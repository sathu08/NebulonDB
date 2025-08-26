from db.index_manager import SegmentManager

import polars as pl
from typing import Union, List
from sentence_transformers import SentenceTransformer

segment_manager = SegmentManager("default")


# Example usage
segment_dataset = {
    "text_col1": ["Hello world", "AI is amazing", "null", "Polars is fast"],
    "text_col2": ["Test sentence", "null", "Another text", "Segment_dataset science"],
    "numeric_col": [1, 2, 3, 4]
}
segment_dataset = {
    "text_col1": ["call boy","mark"],
    "text_col2": ["mental","fellow"],
    "numeric_col": [1, 2]
}

print(segment_manager.insert_segment(segment_dataset=segment_dataset,
                               segment_name="sample",
                               set_columns=["text_col1","text_col2"]))

# Specific column
# print(load_segment(Segment_dataset,segment_name="sample", set_column_vector=["text_col1","text_col2"]))
# print(load_segment(Segment_dataset,segment_name="sample", set_column_vector=["numeric_col"]))

# # All text columns
# print(load_segment(Segment_dataset, set_column_vector="all"))
# print(load_segment(Segment_dataset, set_column_vector=["ALL"]))

# query = ["AI is amazing"]
# model = SentenceTransformer("all-MiniLM-L6-v2")
# query_vec = model.encode(query).tolist()

# results = segment_manager.search_vector(vector=query_vec, top_k=2)
# print(results)


