#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <iostream>
#include <chrono>
#include <thread>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <openssl/evp.h>
#include <openssl/pem.h>

#include <sys/socket.h>

#include "csv_writer.hpp"
#include "market_maker.hpp"
#include "kalshi_auth.hpp"

namespace beast     = boost::beast;
namespace http      = beast::http;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
namespace ssl       = net::ssl;
using tcp           = net::ip::tcp;

// ---------------------------------------------------------------------------
// KalshiWorker
// ---------------------------------------------------------------------------
class KalshiWorker {
public:
    KalshiWorker(const std::string& key_id,
                 const std::string& pem_path,
                 const std::string& ticker,
                 std::shared_ptr<CsvWriter> csv,
                 std::shared_ptr<MarketMaker> mm = nullptr)
        : key_id_(key_id), ticker_(ticker), csv_(csv), mm_(mm), stop_(false)
    {
        // Load private key once
        FILE* f = fopen(pem_path.c_str(), "r");
        if (!f) throw std::runtime_error("Cannot open PEM: " + pem_path);
        pkey_ = PEM_read_PrivateKey(f, nullptr, nullptr, nullptr);
        fclose(f);
        if (!pkey_) throw std::runtime_error("Failed to parse PEM key");
    }

    ~KalshiWorker() {
        if (pkey_) EVP_PKEY_free(pkey_);
    }

    void stop() {
        stop_ = true;
        // Interrupt the blocking ws.read() by shutting down the socket.
        // After resolution Kalshi goes quiet; without this the join blocks forever
        // and save_results() never runs.
        int fd = native_fd_.load();
        if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
    }

    // Blocking run — call from a std::thread
    void run() {
        const std::string host = "api.elections.kalshi.com";
        const std::string port = "443";
        const std::string ws_path = "/trade-api/ws/v2";

        while (!stop_) {
            try {
                net::io_context ioc;
                ssl::context ssl_ctx{ssl::context::tlsv12_client};
                ssl_ctx.set_default_verify_paths();
                ssl_ctx.set_verify_mode(ssl::verify_peer);

                tcp::resolver resolver(ioc);
                auto results = resolver.resolve(host, port);

                beast::ssl_stream<tcp::socket> sock(ioc, ssl_ctx);
                SSL_set_tlsext_host_name(sock.native_handle(), host.c_str());
                net::connect(beast::get_lowest_layer(sock), results);
                native_fd_.store(beast::get_lowest_layer(sock).native_handle());
                sock.handshake(ssl::stream_base::client);

                websocket::stream<beast::ssl_stream<tcp::socket>> ws(std::move(sock));

                // Build auth headers
                std::string packed = kalshi_auth_packed(pkey_, "GET", ws_path);
                auto sep = packed.find('|');
                std::string ts  = packed.substr(0, sep);
                std::string sig = packed.substr(sep + 1);

                ws.set_option(websocket::stream_base::decorator(
                    [&](websocket::request_type& req) {
                        req.set("KALSHI-ACCESS-KEY",       key_id_);
                        req.set("KALSHI-ACCESS-SIGNATURE", sig);
                        req.set("KALSHI-ACCESS-TIMESTAMP", ts);
                        req.set(http::field::user_agent,   "kalshi-collector/1.0");
                    }));

                ws.handshake(host, ws_path);

                // Subscribe to orderbook deltas
                std::string sub_book = R"({"id":1,"cmd":"subscribe","params":{"channels":["orderbook_delta"],"market_ticker":")"
                                       + ticker_ + R"("}})";
                ws.write(net::buffer(sub_book));

                // Subscribe to trade feed (same market, separate id).
                std::string sub_trade = R"({"id":2,"cmd":"subscribe","params":{"channels":["trade"],"market_ticker":")"
                                        + ticker_ + R"("}})";
                ws.write(net::buffer(sub_trade));

                // Subscribe to personal fill notifications for live order tracking.
                // In paper mode on_fill() is a no-op; subscribing is harmless.
                std::string sub_fill = R"({"id":3,"cmd":"subscribe","params":{"channels":["fill"],"market_ticker":")"
                                       + ticker_ + R"("}})";
                ws.write(net::buffer(sub_fill));

                std::cout << "[Kalshi:" << ticker_ << "] connected\n";

                beast::flat_buffer buf;
                while (!stop_) {
                    beast::error_code ec;
                    ws.read(buf, ec);
                    if (ec) {
                        std::cerr << "[Kalshi:" << ticker_ << "] read error: " << ec.message() << "\n";
                        break;
                    }
                    std::string raw = beast::buffers_to_string(buf.data());
                    buf.consume(buf.size());
                    csv_->write_line(raw);

                    // Feed the market maker (paper simulator) if attached.
                    if (mm_) {
                        try {
                            auto jv  = bj::parse(raw);
                            auto& obj = jv.as_object();
                            auto* tv = obj.if_contains("type");
                            auto* mv = obj.if_contains("msg");
                            if (tv && mv && mv->is_object()) {
                                std::string type = kalshi::parse_str(*tv);
                                auto& msg = mv->as_object();
                                double ts = std::chrono::duration<double>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();
                                if (type == "orderbook_snapshot")
                                    mm_->on_snapshot(msg);
                                else if (type == "orderbook_delta")
                                    mm_->on_delta(msg, ts);
                                else if (type == "fill")
                                    mm_->on_fill(msg);
                            }
                        } catch (...) {}
                    }
                }

                native_fd_.store(-1);
                beast::error_code ec;
                if (!stop_) ws.close(websocket::close_code::normal, ec);

            } catch (const std::exception& e) {
                native_fd_.store(-1);
                if (!stop_) {
                    std::cerr << "[Kalshi:" << ticker_ << "] exception: " << e.what()
                              << " — retrying in 2s\n";
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
        }
        std::cout << "[Kalshi:" << ticker_ << "] worker stopped\n";
    }

private:
    std::string key_id_;
    std::string ticker_;
    std::shared_ptr<CsvWriter>    csv_;
    std::shared_ptr<MarketMaker>  mm_;
    std::atomic<bool> stop_;
    std::atomic<int>  native_fd_{-1};
    EVP_PKEY* pkey_ = nullptr;
};
