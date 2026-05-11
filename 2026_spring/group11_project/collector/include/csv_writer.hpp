#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <filesystem>

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) : path_(path) {
        std::filesystem::create_directories(
            std::filesystem::path(path).parent_path());
        file_.open(path, std::ios::app);
        if (!file_.is_open())
            throw std::runtime_error("Cannot open CSV: " + path);
    }

    // Write one raw JSON line. Strips any trailing \r or \n from the
    // source (WebSocket frames often end with \r\n) then appends exactly
    // one newline, so the file never contains blank lines.
    void write_line(const std::string& raw_json) {
        std::lock_guard<std::mutex> lock(mu_);
        std::size_t end = raw_json.size();
        while (end > 0 && (raw_json[end-1] == '\r' || raw_json[end-1] == '\n'))
            --end;
        if (end == 0) return;  // skip empty/whitespace-only frames
        file_.write(raw_json.data(), (std::streamsize)end);
        file_ << "\n";
        file_.flush();
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::ofstream file_;
    std::mutex mu_;
};