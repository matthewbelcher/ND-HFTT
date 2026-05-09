#include <string>
#include <ctime>

#define BASE_URL "https://www.federalreserve.gov/newsevents/pressreleases/"

static const std::string releaseDates[] = {
    "0507", // May 6-7
    "0618", // June 17-18*
    "0730", // July 29-30
    "0917", // September 16-17*
    "1029", // October 28-29
    "1210"  // December 9-10*
};

std::string formatDateWithCurrentYear(const std::string& date) {
    // Get the current year
    time_t now = time(0);
    tm* localTime = localtime(&now);
    int currentYear = 1900 + localTime->tm_year;

    // Format the string
    std::string formattedDate = "monetary" + std::to_string(currentYear) + date + "a.htm";
    return formattedDate;
}