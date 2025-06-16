from src.pipeline.model_pipeline import review_score_prediction_pipeline

if __name__ == "__main__":
    review_score_prediction_pipeline(data_path="./data/olist_order_reviews_dataset.csv")
