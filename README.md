# Warc-modified

## This repo holds a modification of the WARC-dl pipeline

general modification:
- pipeline.py:
  - modified constructor to pass through texts and meta information without prediciton

Use Cases:

Twitter extraction:
- extracting twitter html from warc files, obtaining text and meta information
twitter_text_pipeline.py
  - modifications:
    - inserted search terms
    - yield text, timestamp, url, http-header
    - fitted get_signature() tensor specs
    - export into single csv file
  
  
Blogspot extraction:

- extracting blogspot html from warc files, obtaining text and meta information
blogspot_text_pipeline.py
  - modifications:
    - check for csv file in constructor
    - inserted search terms
    - added timestamp extraction logic
    - yield text, timestamp, url, comment-tag
    - fitted get_signature() tensor specs
    - export into single csv file
