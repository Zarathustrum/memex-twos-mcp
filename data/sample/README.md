# Sample Data

This folder contains a sanitized Twos markdown export for testing.

## Files

- `sample_export.md` - Generic Twos export (Markdown with timestamps format) with placeholder names and tags.

## Usage

1. Copy the sample export into the raw data folder:
   ```bash
   cp data/sample/sample_export.md data/raw/twos_export.md
   ```
2. Convert to JSON:
   ```bash
   python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json --pretty
   ```
3. Load into SQLite:
   ```bash
   python3 scripts/load_to_sqlite.py data/processed/twos_data.json
   ```

This sample data is safe for public demos and does not contain personal information.
