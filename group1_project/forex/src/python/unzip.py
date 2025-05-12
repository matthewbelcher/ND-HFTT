import os
import subprocess
import argparse

def break_hard_link(file_path):
    """
    Breaks the hard link by copying the file to a temporary file and moving it back.
    This creates a new inode for the file.
    """
    tmp_file = file_path + ".tmp"
    subprocess.run(['cp', '-p', file_path, tmp_file], check=True)
    subprocess.run(['mv', tmp_file, file_path], check=True)

def process_files_in_directory(root_dir, mode):
    """
    Recursively walk through root_dir and:
      - if mode == 'gunzip': decompress files ending with .gz
      - if mode == 'gzip': compress files that are not already compressed.
        For files with multiple hard links, break the link first.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if mode == "gunzip" and filename.endswith('.gz'):
                # Check for multiple hard links before decompressing.
                link_count = os.stat(file_path).st_nlink
                if link_count > 1:
                    print(f"Breaking hard link for {file_path} (link count {link_count})")
                    break_hard_link(file_path)
                print(f"Gunzipping: {file_path}")
                subprocess.run(['gunzip', file_path], check=True)
            elif mode == "gzip" and not filename.endswith('.gz'):
                link_count = os.stat(file_path).st_nlink
                if link_count > 1:
                    print(f"Breaking hard link for {file_path} (link count {link_count})")
                    break_hard_link(file_path)
                print(f"Gzipping: {file_path}")
                subprocess.run(['gzip', file_path], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively process files in a directory in gzip or gunzip mode."
    )
    parser.add_argument("root_dir", help="Root directory to process files.")
    parser.add_argument(
        "mode",
        choices=["gzip", "gunzip"],
        help="Operation mode: 'gzip' to compress files, 'gunzip' to decompress .gz files."
    )
    args = parser.parse_args()

    process_files_in_directory(args.root_dir, args.mode)
