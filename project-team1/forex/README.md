## C++ Library Compilation

Ensure `fmt` is installed. For example, on macOS with Homebrew, run the following:

```
brew install fmt
```

To compile the library and all binaries, run the following:

```
make
```

The `build/` directory will contain the library and all executables.

## Data 

We used proprietary forex market data provided by Polygon.io. The complete dataset size amounts to multiple dozens of gigabytes. We provided a small sample of filtered data in the `data/` directory.
