#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>
#include <sstream>
#include <iomanip>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ec.h>

#include <sys/socket.h>

#include "csv_writer.hpp"
#include "market_maker.hpp"

namespace beast     = boost::beast;
namespace http      = beast::http;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
namespace ssl       = net::ssl;
using tcp           = net::ip::tcp;

// ---------------------------------------------------------------------------
// Minimal JWT builder for ES256 (Coinbase CDP keys)
// ---------------------------------------------------------------------------

// URL-safe base64 (no padding) for JWT
inline std::string b64url(const unsigned char* data, size_t len) {
    static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < len; i += 3) {
        unsigned int v = (unsigned char)data[i] << 16;
        if (i+1 < len) v |= (unsigned char)data[i+1] << 8;
        if (i+2 < len) v |= (unsigned char)data[i+2];
        out += b64[(v>>18)&63]; out += b64[(v>>12)&63];
        out += (i+1<len) ? b64[(v>>6)&63] : '=';
        out += (i+2<len) ? b64[v&63]      : '=';
    }
    // Convert to URL-safe, strip padding
    for (auto& c : out) { if (c=='+') c='-'; if (c=='/') c='_'; }
    while (!out.empty() && out.back()=='=') out.pop_back();
    return out;
}

inline std::string b64url_str(const std::string& s) {
    return b64url(reinterpret_cast<const unsigned char*>(s.data()), s.size());
}

inline std::string hex_nonce(size_t bytes = 16) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 255);
    std::ostringstream oss;
    for (size_t i = 0; i < bytes; i++)
        oss << std::hex << std::setw(2) << std::setfill('0') << dist(gen);
    return oss.str();
}

inline std::string build_coinbase_jwt(const std::string& api_key,
                                       EVP_PKEY* pkey) {
    long now = (long)std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Header
    std::string header_json = R"({"alg":"ES256","kid":")" + api_key + R"(","nonce":")" + hex_nonce() + R"("})";
    std::string header = b64url_str(header_json);

    // Payload
    std::string payload_json = R"({"iss":"cdp","sub":")" + api_key
        + R"(","nbf":)" + std::to_string(now)
        + R"(,"exp":)" + std::to_string(now + 120) + "}";
    std::string payload = b64url_str(payload_json);

    std::string signing_input = header + "." + payload;

    // Sign with ES256
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestSignInit(ctx, nullptr, EVP_sha256(), nullptr, pkey);
    EVP_DigestSignUpdate(ctx, signing_input.data(), signing_input.size());
    size_t sig_len = 0;
    EVP_DigestSignFinal(ctx, nullptr, &sig_len);
    std::vector<unsigned char> der_sig(sig_len);
    EVP_DigestSignFinal(ctx, der_sig.data(), &sig_len);
    EVP_MD_CTX_free(ctx);

    // DER-encoded ECDSA sig → raw R||S (64 bytes for P-256)
    const unsigned char* ptr = der_sig.data();
    ECDSA_SIG* ecdsa = d2i_ECDSA_SIG(nullptr, &ptr, (long)sig_len);
    const BIGNUM *r, *s;
    ECDSA_SIG_get0(ecdsa, &r, &s);
    std::vector<unsigned char> raw_sig(64, 0);
    BN_bn2binpad(r, raw_sig.data(),      32);
    BN_bn2binpad(s, raw_sig.data() + 32, 32);
    ECDSA_SIG_free(ecdsa);

    std::string sig_b64 = b64url(raw_sig.data(), 64);
    return signing_input + "." + sig_b64;
}

// ---------------------------------------------------------------------------
// CoinbaseWorker
// ---------------------------------------------------------------------------
class CoinbaseWorker {
public:
    CoinbaseWorker(const std::string& api_key,
                   const std::string& private_key_pem,   // PEM string (not path)
                   std::shared_ptr<CsvWriter> csv,
                   std::shared_ptr<MarketMaker> mm = nullptr)
        : api_key_(api_key), csv_(csv), mm_(mm), stop_(false)
    {
        // Parse EC private key from PEM string
        BIO* bio = BIO_new_mem_buf(private_key_pem.data(), (int)private_key_pem.size());
        pkey_ = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
        BIO_free(bio);
        if (!pkey_) throw std::runtime_error("Failed to parse Coinbase EC key from JSON");
    }

    ~CoinbaseWorker() {
        if (pkey_) EVP_PKEY_free(pkey_);
    }

    void stop() {
        stop_ = true;
        int fd = native_fd_.load();
        if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
    }

    void run() {
        const std::string host = "advanced-trade-ws.coinbase.com";
        const std::string port = "443";
        const std::string path = "/";

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
                ws.set_option(websocket::stream_base::decorator(
                    [](websocket::request_type& req) {
                        req.set(http::field::user_agent, "kalshi-collector/1.0");
                    }));
                ws.handshake(host, path);

                // Subscribe with fresh JWT.
                // "ticker"        — fires on every best-bid/ask change and trade; ~5-15 MB/hr
                // "market_trades" — completed trades only; ~10-30 MB/hr
                // Together these replace "level2" (~270 MB/hr) at ~5% of the volume.
                std::string jwt = build_coinbase_jwt(api_key_, pkey_);
                auto cb_subscribe = [&](const std::string& channel) {
                    std::string sub = R"({"type":"subscribe","product_ids":["BTC-USD"],"channel":")"
                                      + channel + R"(","jwt":")" + jwt + R"("})";
                    ws.write(net::buffer(sub));
                };
                cb_subscribe("ticker");
                cb_subscribe("market_trades");

                std::cout << "[Coinbase] connected\n";

                beast::flat_buffer buf;
                while (!stop_) {
                    beast::error_code ec;
                    ws.read(buf, ec);
                    if (ec) {
                        std::cerr << "[Coinbase] read error: " << ec.message() << "\n";
                        break;
                    }
                    std::string raw = beast::buffers_to_string(buf.data());
                    buf.consume(buf.size());
                    csv_->write_line(raw);

                    // Extract BTC mid price from ticker channel for MM BTC momentum.
                    // Ticker fires on every best-bid/ask change — fine resolution for 10s momentum.
                    if (mm_) {
                        try {
                            auto jv  = bj::parse(raw);
                            auto& obj = jv.as_object();
                            auto* cv = obj.if_contains("channel");
                            auto* ev = obj.if_contains("events");
                            if (cv && ev && ev->is_array() &&
                                kalshi::parse_str(*cv) == "ticker") {
                                double ts = std::chrono::duration<double>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();
                                for (auto& event : ev->as_array()) {
                                    if (!event.is_object()) continue;
                                    auto* tv = event.as_object().if_contains("tickers");
                                    if (!tv || !tv->is_array()) continue;
                                    for (auto& ticker : tv->as_array()) {
                                        if (!ticker.is_object()) continue;
                                        auto& tk = ticker.as_object();
                                        auto* bv = tk.if_contains("best_bid");
                                        auto* av = tk.if_contains("best_ask");
                                        if (!bv || !av) continue;
                                        double bid = kalshi::parse_num(*bv);
                                        double ask = kalshi::parse_num(*av);
                                        if (bid > 0.0 && ask > 0.0)
                                            mm_->add_btc_price(ts, (bid + ask) / 2.0);
                                    }
                                }
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
                    std::cerr << "[Coinbase] exception: " << e.what()
                              << " — retrying in 2s\n";
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
        }
        std::cout << "[Coinbase] worker stopped\n";
    }

private:
    std::string api_key_;
    std::shared_ptr<CsvWriter>   csv_;
    std::shared_ptr<MarketMaker> mm_;
    std::atomic<bool> stop_;
    std::atomic<int>  native_fd_{-1};
    EVP_PKEY* pkey_ = nullptr;
};