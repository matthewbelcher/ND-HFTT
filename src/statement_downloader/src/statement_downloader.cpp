#include "statement_downloader/statement_downloader.hpp"

const std::string not_found_string = "Page not Found";
const std::string report_format_string = "<div class=\"col-xs-12 col-sm-8 col-md-8\">";
const std::string federal_funds_rate_string = "federal funds rate";

double parseMixedNumber(const char* str) {
    int whole = 0;
    int numerator = 0;
    int denominator = 1;

    if (*(str + 1) == '-') {
        if (sscanf(str, "%d-%d/%d", &whole, &numerator, &denominator) > 0) {
            if (numerator == 0) {
                return whole;
            } else {
                return whole + ((double) numerator / denominator);
            }
        } else {
            return -1;
        }
    } else if (*(str + 1) < 0) {
        if (sscanf(str, "%d‑%d/%d", &whole, &numerator, &denominator) > 0) {
            if (numerator == 0) {
                return whole;
            } else {
                return whole + ((double) numerator / denominator);
            }
        } else {
            return -1;
        }
    } else {
        if (sscanf(str, "%d/%d", &numerator, &denominator) > 0) {
            if (numerator == 0) {
                return 0;
            } else {
                return ((double) numerator / denominator);
            }
        } else {
            return -1;
        }
    }
}

size_t WriteMemoryCallback(void *ptr, size_t size, size_t nmemb, void *userdata) {
    MemoryStruct *mem = static_cast<MemoryStruct*>(userdata);

    size_t realsize = size * nmemb;

    if(mem->length + realsize > mem->size) {
        printf("Not enough memory to store the data. Current length: %zu, trying to add: %zu\n", mem->length, realsize);
        mem->memory = static_cast<char*>(realloc(mem->memory, mem->size * 2));
        if(mem->memory == NULL) {
            fprintf(stderr, "not enough memory (realloc returned NULL)\n");
            return 0;
        }
        mem->size = mem->size * 2;
    }

    if (mem->length == 0) {
        char* match = std::search(static_cast<char*>(ptr), static_cast<char*>(ptr) + realsize, not_found_string.begin(), not_found_string.end());
        if (match < static_cast<char*>(ptr) + realsize) {
            printf("Statement Not Found\n");
            return 0;
        }
    }

    memcpy(&(mem->memory[mem->length]), ptr, realsize);
    mem->length += realsize;
    mem->memory[mem->length] = 0;

    return realsize;
}

StatementDownloader::StatementDownloader(std::string _statement_url, std::chrono::time_point<std::chrono::system_clock> _release_time, size_t _initial_mem_size){
    statement_url = _statement_url;
    release_time = _release_time;

    chunk.memory = static_cast<char*>(malloc(_initial_mem_size));
    if (chunk.memory == NULL) {
        fprintf(stderr, "Not enough memory (malloc returned NULL)\n");
        exit(EXIT_FAILURE);
    }
    chunk.length = 0;
    chunk.size = _initial_mem_size;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle = curl_easy_init();

    if(curl_handle) {
        curl_easy_setopt(curl_handle, CURLOPT_URL, statement_url.c_str());
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
        curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    }
}

StatementDownloader::~StatementDownloader(){
    if (chunk.memory) {
        free(chunk.memory);
    }

    if (curl_handle) {
        curl_easy_cleanup(curl_handle);
    }

    curl_global_cleanup();
}

std::pair<double, double> StatementDownloader::getRateBounds() const {
    return std::make_pair(lower_bound, upper_bound);
}

int StatementDownloader::download() {
    if (!curl_handle) {
        fprintf(stderr, "CURL handle is not initialized.\n");
        return -1;
    }

    while(true) {
        if(release_time > std::chrono::system_clock::now() + std::chrono::seconds(3)) {
            break;
        }
    }

    while (true) {
        CURLcode res = curl_easy_perform(curl_handle);

        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            printf("%lu bytes retrieved\n", (long)chunk.length);
            return 0;
        }

        reset();
    }
}

int StatementDownloader::parse() {
    char* match = std::search(static_cast<char*>(chunk.memory), static_cast<char*>(chunk.memory) + chunk.length, report_format_string.begin(), report_format_string.end());
    if (match >= static_cast<char*>(chunk.memory) + chunk.length) {
        printf("Paragraph Not Found\n");
        return -1;
    }

    char* match2 = std::search(match, static_cast<char*>(chunk.memory) + chunk.length, federal_funds_rate_string.begin(), federal_funds_rate_string.end());
    if (match2 >= static_cast<char*>(chunk.memory) + chunk.length) {
        printf("Federal Funds Rate Not Found\n");
        return -1;
    }
    match2 += 18;

    int upper_offset = -1;
    int lower_offset = -1;

    while (true) {

        int i = 0;
        int phase = 0;
        while (i < 50) {
            if (phase == 0) {
                if (*(match2 + i) >= '0' && *(match2 + i) <= '9') {
                    phase = 1;
                    lower_offset = i;
                }
            } else if (phase == 1) {
                if (*(match2 + i) == ' ') {
                    phase = 2;
                }
            } else if (phase == 2) {
                if (strncmp(match2 + i, "to ", 3) == 0) {
                    phase = 3;
                } else {
                    phase = 0;
                }
            } else if (phase == 3) {
                if (*(match2 + i) > '0' && *(match2 + i) < '9') {
                    upper_offset = i;
                    break;
                }
            }

            i++;
        }

        if (upper_offset != -1 && lower_offset != -1) {
            break;
        }

        match2 = std::search(match2, static_cast<char*>(chunk.memory) + chunk.length, federal_funds_rate_string.begin(), federal_funds_rate_string.end());
        if (match2 >= static_cast<char*>(chunk.memory) + chunk.length) {
            printf("Federal Funds Rate Not Found\n");
            return -1;
        }
        match2 += 18;
    }
    
    lower_bound = parseMixedNumber(match2 + lower_offset);
    if (lower_bound == -1) {
        printf("Invalid Mixed Number for Lower Bound\n");
        return -1;
    }

    upper_bound = parseMixedNumber(match2 + upper_offset);
    if (upper_bound == -1) {
        printf("Invalid Mixed Number for Upper Bound\n");
        return -1;
    }

    return 0;
}

inline void StatementDownloader::reset() {
    chunk.length = 0;
}