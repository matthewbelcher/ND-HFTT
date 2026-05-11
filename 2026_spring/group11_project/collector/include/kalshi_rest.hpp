#pragma once

#include <string>
#include <stdexcept>
#include <iostream>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/json.hpp>

#include <openssl/evp.h>
#include <openssl/pem.h>

#include "kalshi_auth.hpp"

namespace bj = boost::json;

// ---------------------------------------------------------------------------
// KalshiRestClient
//
// Synchronous Boost.Beast HTTPS client for Kalshi order management.
// Each call opens a new TLS connection (keep-alive not needed at this rate).
//
// All methods are blocking and NOT thread-safe: the caller (MarketMaker)
// already holds its mutex when calling into here, so no additional locking
// is needed.
// ---------------------------------------------------------------------------
class KalshiRestClient {
public:
    static constexpr const char* HOST    = "api.elections.kalshi.com";
    static constexpr const char* PORT    = "443";
    static constexpr const char* API_PFX = "/trade-api/v2";

    KalshiRestClient(const std::string& key_id, const std::string& pem_path)
        : key_id_(key_id)
    {
        FILE* f = fopen(pem_path.c_str(), "r");
        if (!f) throw std::runtime_error("KalshiRestClient: cannot open PEM: " + pem_path);
        pkey_ = PEM_read_PrivateKey(f, nullptr, nullptr, nullptr);
        fclose(f);
        if (!pkey_) throw std::runtime_error("KalshiRestClient: failed to parse PEM key");
    }

    ~KalshiRestClient() {
        if (pkey_) EVP_PKEY_free(pkey_);
    }

    // Place a maker limit order.
    // action : "buy" or "sell"
    // side   : "yes" or "no"
    // price_cents : 1-99 (yes_price when side=yes, no_price when side=no)
    // qty    : number of contracts
    // post_only : true for maker-only (rejected if it would cross)
    // tif    : "good_till_canceled" | "immediate_or_cancel" | "fill_or_kill"
    // Returns order_id string on success, empty string on error.
    std::string place_order(const std::string& ticker,
                            const std::string& action,
                            const std::string& side,
                            int price_cents,
                            int qty,
                            bool post_only = true,
                            const std::string& tif = "good_till_canceled") {
        bj::object body;
        body["ticker"]           = ticker;
        body["action"]           = action;
        body["side"]             = side;
        body["count"]            = qty;
        body["type"]             = "limit";
        body["time_in_force"]    = tif;
        if (post_only) body["post_only"] = true;
        if (side == "yes")
            body["yes_price"] = price_cents;
        else
            body["no_price"]  = price_cents;

        std::string path = std::string(API_PFX) + "/portfolio/orders";
        std::string resp = do_request("POST", path, bj::serialize(body));
        if (resp.empty()) return {};

        try {
            auto jv  = bj::parse(resp);
            auto& jo = jv.as_object();
            auto* ov = jo.if_contains("order");
            if (!ov || !ov->is_object()) return {};
            auto& ord = ov->as_object();
            auto* iv  = ord.if_contains("order_id");
            if (!iv) return {};
            return std::string(iv->as_string());
        } catch (...) {
            return {};
        }
    }

    // Cancel (fully reduce) a resting order.
    // Returns true (cancelled, HTTP 200) or false (error).
    // Sets was_filled=true when HTTP 404 — order was already filled, not just missing.
    bool cancel_order(const std::string& order_id, bool& was_filled) {
        was_filled = false;
        if (order_id.empty()) return false;
        std::string path = std::string(API_PFX) + "/portfolio/orders/" + order_id;
        int status = 0;
        do_request("DELETE", path, "", &status);
        if (status == 200) return true;
        if (status == 404) was_filled = true;
        return false;
    }

private:
    std::string key_id_;
    EVP_PKEY*   pkey_ = nullptr;

    // status_out (optional): set to HTTP response status code on return.
    std::string do_request(const std::string& method,
                           const std::string& path,
                           const std::string& body,
                           int* status_out = nullptr) {
        namespace beast = boost::beast;
        namespace http  = beast::http;
        namespace net   = boost::asio;
        namespace ssl   = net::ssl;
        using tcp = net::ip::tcp;

        try {
            net::io_context ioc;
            ssl::context ssl_ctx{ssl::context::tlsv12_client};
            ssl_ctx.set_default_verify_paths();
            ssl_ctx.set_verify_mode(ssl::verify_peer);

            tcp::resolver resolver(ioc);
            auto endpoints = resolver.resolve(HOST, PORT);

            beast::ssl_stream<tcp::socket> stream(ioc, ssl_ctx);
            SSL_set_tlsext_host_name(stream.native_handle(), HOST);
            net::connect(beast::get_lowest_layer(stream), endpoints);
            stream.handshake(ssl::stream_base::client);

            // Auth
            std::string packed = kalshi_auth_packed(pkey_, method, path);
            auto sep = packed.find('|');
            std::string ts  = packed.substr(0, sep);
            std::string sig = packed.substr(sep + 1);

            http::verb verb;
            if      (method == "POST")   verb = http::verb::post;
            else if (method == "DELETE") verb = http::verb::delete_;
            else if (method == "GET")    verb = http::verb::get;
            else throw std::runtime_error("unknown HTTP method");

            http::request<http::string_body> req{verb, path, 11};
            req.set(http::field::host,         HOST);
            req.set(http::field::user_agent,   "kalshi-collector/1.0");
            req.set(http::field::content_type, "application/json");
            req.set("KALSHI-ACCESS-KEY",        key_id_);
            req.set("KALSHI-ACCESS-SIGNATURE",  sig);
            req.set("KALSHI-ACCESS-TIMESTAMP",  ts);
            if (!body.empty()) {
                req.body() = body;
                req.prepare_payload();
            }

            http::write(stream, req);

            beast::flat_buffer buf;
            http::response<http::string_body> res;
            http::read(stream, buf, res);

            beast::error_code ec;
            stream.shutdown(ec);  // best-effort TLS shutdown

            int status = static_cast<int>(res.result_int());
            if (status_out) *status_out = status;
            if (status >= 400) {
                std::cerr << "[REST] " << method << " " << path
                          << " → HTTP " << status << ": " << res.body() << "\n";
                return {};
            }
            return res.body();

        } catch (const std::exception& e) {
            if (status_out) *status_out = 0;
            std::cerr << "[REST] " << method << " " << path
                      << " exception: " << e.what() << "\n";
            return {};
        }
    }
};
