# Exact Commands to Run - Step by Step

## Step 1: Create directory and copy files
```bash
mkdir -p ~/Desktop/transcripts_to_process
cp -r ~/Documents/subtitles-DrBoz/* ~/Desktop/transcripts_to_process/
```

## Step 2: Go to our project directory
```bash
cd /Users/markmacmini/Documents/Kilo-Code
```

## Step 3: Process the files
```bash
python3 batch_processor.py --input-dir ~/Desktop/transcripts_to_process --output-file output/drboz_results.jsonl
```

## Alternative: Use our custom script
```bash
cd /Users/markmacmini/Documents/Kilo-Code
python3 process_your_files.py
```

## If you want to use the main script (simpler)
```bash
cd /Users/markmacmini/Documents/Kilo-Code
cp -r ~/Desktop/transcripts_to_process/* data/transcripts/
python3 main.py
```

## Troubleshooting:
- Make sure you're in the Kilo-Code directory when running python commands
- Use `python3` not `python`
- Check that the files were copied correctly with `ls ~/Desktop/transcripts_to_process`