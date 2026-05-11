#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>

#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

// ---------------------------------------------------------------------------
// RSA-PSS auth helpers shared by KalshiWorker and KalshiRestClient
// ---------------------------------------------------------------------------

inline std::string base64_encode(const unsigned char* data, size_t len) {
    static const char* b64 =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned int val = data[i] << 16;
        if (i+1 < len) val |= data[i+1] << 8;
        if (i+2 < len) val |= data[i+2];
        out += b64[(val >> 18) & 63];
        out += b64[(val >> 12) & 63];
        out += (i+1 < len) ? b64[(val >>  6) & 63] : '=';
        out += (i+2 < len) ? b64[ val        & 63] : '=';
    }
    return out;
}

inline std::string sign_pss(EVP_PKEY* pkey, const std::string& message) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_PKEY_CTX* pkey_ctx = nullptr;

    if (EVP_DigestSignInit(ctx, &pkey_ctx, EVP_sha256(), nullptr, pkey) <= 0)
        throw std::runtime_error("EVP_DigestSignInit failed");

    EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, RSA_PKCS1_PSS_PADDING);
    EVP_PKEY_CTX_set_rsa_pss_saltlen(pkey_ctx, 32);
    EVP_PKEY_CTX_set_rsa_mgf1_md(pkey_ctx, EVP_sha256());

    EVP_DigestSignUpdate(ctx, message.data(), message.size());

    size_t sig_len = 0;
    EVP_DigestSignFinal(ctx, nullptr, &sig_len);
    std::vector<unsigned char> sig(sig_len);
    EVP_DigestSignFinal(ctx, sig.data(), &sig_len);
    EVP_MD_CTX_free(ctx);

    return base64_encode(sig.data(), sig_len);
}

// Builds the three Kalshi auth header values and returns them packed.
// Caller splits on '|': ts = packed.substr(0, packed.find('|'))
//                        sig = packed.substr(packed.find('|') + 1)
inline std::string kalshi_auth_packed(EVP_PKEY* pkey,
                                      const std::string& method,
                                      const std::string& path_no_query) {
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string ts  = std::to_string(now_ms);
    std::string sig = sign_pss(pkey, ts + method + path_no_query);
    return ts + "|" + sig;
}
