from kalshi_culture_analyzer.ticker_parser import parse_ticker_file


def test_parse_ticker_file(tmp_path):
    content = """
# comment line
// another comment
KXSURVIVOR-26DEC31 # reality tv
KXOTHER-25JAN01 // inline comment

KXSHOW-27FEB27
"""
    path = tmp_path / "tickers.txt"
    path.write_text(content, encoding="utf-8")
    entries = parse_ticker_file(str(path))
    assert len(entries) == 3
    assert entries[0].ticker == "KXSURVIVOR-26DEC31"
    assert "reality" in entries[0].tags
    assert entries[1].ticker == "KXOTHER-25JAN01"
    assert entries[2].ticker == "KXSHOW-27FEB27"

