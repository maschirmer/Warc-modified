import abc
import os
from collections import Counter
import re
import csv
from dateutil.parser import parse
import json




import tensorflow as tf
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree

from helpers import create_s3_client, get_file_stream
from pipelines.pipeline import Pipeline


class BlogPipeline(Pipeline, abc.ABC):
    """
    This pipeline extracts texts from websites from the WARC files. It streams the following to the driver/GPU:
    An (optionally tokenized) version of the website text, which should be as clean as possible (useful for neural
    network input),
    an original version of the text as a string,
    the website url.
    """

    def __init__(self, out_dir, max_content_length):
        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        self.max_content_length = max_content_length
        

        self.csv_out = f"{self.out_dir}/blogs_large_commoncrawl.csv"
        

        if not os.path.exists(self.csv_out):
            with open(self.csv_out, mode='w', encoding='utf-8', newline="\n") as f:
                writer = csv.writer(f)
                writer.writerow(["text","url","date","comment"])
        
        super().__init__()



    def get_signature(self):
        return (
            tf.TensorSpec(shape=(), dtype=tf.string), # export text
            tf.TensorSpec(shape=(), dtype=tf.string),  # url
            tf.TensorSpec(shape=(), dtype=tf.string),   # date timestamp
            tf.TensorSpec(shape=(), dtype=tf.string))    # comment

    def get_distributed_filter(self):
        """
        Overridable method that provides a filter, which is executed on the pyspark cluster nodes.
        The returned distributed_filter must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of distributed_filter, which may then use these variables.
        """

        def distributed_filter(text):
            return True

        return distributed_filter

    def get_tokens_spec(self):
        """
        Overridable method that returns a tf.TensorSpec which corresponds to the values returned by the tokenizer
        defined in get_tokenizer().
        """

        return tf.TensorSpec(shape=(), dtype=tf.string)

    def get_tokenizer(self):
        """
        Overridable method that provides a tokenizer, which is executed on the pyspark cluster nodes.
        The returned tokenizer must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of tokenizer, which may then use these variables.
        """

        def tokenizer(text):
            return text

        return tokenizer

    def get_generator_factory(self):
        
        acc_counter = self.acc_counter
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        #tokenizer = self.get_tokenizer()
        
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL

        def generator_factory(file_identifier):
            try:
                s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
                
                stream = get_file_stream(s3_client, file_identifier)
                
                for record in ArchiveIterator(stream, max_content_length=max_content_length):
                    
                    try:
                        
                        if record.headers is None:
                            # empty header
                            acc_counter.add(Counter({"n_record_headers_none": 1}))
                            continue
                        
                        if record.http_headers is None:
                            # no http_header
                            acc_counter.add(Counter({"n_http_headers_none": 1}))
                            continue
                        
                        if record.headers['WARC-Type'] == 'response' and record.content_length >= 128:
                            content_type = str(record.http_content_type).lower()
                            
                            if content_type.startswith("text/html"):
                                
                                url = str(record.headers['WARC-Target-URI'])

                                
                                if re.findall("blogspot.com/\d{4}/\d{2}/", url) != []:  
                                    
                                    # determine if its a comments html
                                    if "show" in url and "Comment" in url:
                                        comment = "1"
                                    else:
                                        comment = "0"
                                    

                                    html_bytes = record.reader.read()
                                    
                                    try:
                                        encoding = record.http_charset
                                        if encoding is None:
                                            encoding = detect_encoding(html_bytes)
                                        tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
                                    
                                    except:
                                        acc_counter.add(Counter({"n_parsing_exception": 1}))
                                        continue
                                    

                                    # extract date
                                    try:
                                        p = tree.body.get_elements_by_class_name('date-header')
                                        date = p.query_selector('span').text
                                        date = parse(date).strftime("%d/%m/%Y")

                                    except:
                                        try:
                                            date_strings = re.findall("blogspot.com/\d{4}/\d{2}/", url)[0].split("/")
                                            year = date_strings[1]
                                            month = date_strings[2]
                                            date = f"01/{month}/{year}"
                                        
                                        except:
                                            try:
                                                date = str(record.headers['WARC-Date'])
                                                date = parse(date).strftime("%d/%m/%Y")
                                                acc_counter.add(Counter({"n_used_warcdate": 1}))
                                            except:
                                                date = "01/01/1901"
                                                acc_counter.add(Counter({"n_no_possible_date": 1}))
                                        #         continue
                                            
                                        #     continue
                                        
                                        # continue
                                    
                                    # extract text
                                    export_text = extract_plain_text(tree, preserve_formatting=True, main_content=True,
                                                                list_bullets=False, alt_texts=True, links=False,
                                                                form_fields=False, noscript=True)
                                    
                                    
                                    if not distributed_filter(export_text):

                                        acc_counter.add(Counter({"n_distributed_filter_not_passed": 1}))
                                        continue

                                    yield  export_text, url, date, comment 
                                    acc_counter.add(Counter({"n_node_results": 1}))
                                
                                else:
                                    # blogspot nicht in url enthalten
                                    acc_counter.add(Counter({"n_no_blogspot_url": 1}))
                                    continue

                            else:
                                acc_counter.add(Counter({"n_wrong_content_type": 1}))
                                continue
                        
                        else:
                            acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                            continue
                    
                    except:
                        acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                        continue
                
                ## end of for loop
                
                acc_counter.add(Counter({"n_finished_warc_files": 1}))
            
            except:
                yield  "errortext", "errorurl", "errordate", "errorcomment"
                acc_counter.add(Counter({"n_aws_stream_exception": 1}))

        return generator_factory


    def export(self, export_text, url, date, comment):
        row = [ export_text.decode("utf-8"), url.decode("utf-8"), date.decode("utf-8"), comment.decode("utf-8") ]
        
        print(row[2])
        
        with open(self.csv_out, "a", encoding="utf-8", errors="ignore", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(row)



