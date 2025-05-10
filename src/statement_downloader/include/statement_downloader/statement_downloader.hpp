#ifndef STATEMENT_DOWNLOADER_HPP
#define STATEMENT_DOWNLOADER_HPP

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <curl/curl.h>

struct MemoryStruct {
    char *memory;
    size_t length;
    size_t size;
};


class StatementDownloader {
    private:
        std::string statement_url;
        std::chrono::time_point<std::chrono::system_clock> release_time;
        MemoryStruct chunk;
        CURL *curl_handle;

        double lower_bound;
        double upper_bound;

    public:
        StatementDownloader(std::string statement_url, std::chrono::time_point<std::chrono::system_clock> release_time, size_t initial_mem_size = 80000);
        ~StatementDownloader();

        std::pair<double, double> getRateBounds() const;

        int download();
        int parse();
        void reset();
};


double parseMixedNumber(const char* str);

#endif