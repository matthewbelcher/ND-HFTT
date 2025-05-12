#include "statement_downloader/statement_downloader.hpp"

int main(int argc, char *argv[])
{
    StatementDownloader downloader = StatementDownloader("https://www.federalreserve.gov/newsevents/pressreleases/monetary20250319a.htm", std::chrono::system_clock::now() + std::chrono::seconds(30), 80000);
    
    int result = downloader.download();
    if (result != 0) {
        fprintf(stderr, "Failed to download the statement.\n");
        return result;
    }
    printf("Statement downloaded successfully.\n");

    int rc = downloader.parse();

    return 0;
}