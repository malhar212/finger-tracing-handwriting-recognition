# CS 5330
# Malhar Mahant & Kruthika Gangaraju & Sriram Kodeeswaran
# Final Project: Handwriting gesture detection and recognition
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Class that loads the specified model from file and provides helper method for inference
class MyTrOCRModel:
    def __init__(self, trained_checkpoint="microsoft/trocr-base-handwritten"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trocr_default = "microsoft/trocr-base-handwritten"
        self.processor = TrOCRProcessor.from_pretrained(trocr_default)
        self.model = VisionEncoderDecoderModel.from_pretrained(trained_checkpoint)
        self.model.to(self.device)
        self.model_configurator()

    # Helper method to set model configurations
    def model_configurator(self):
        # set special tokens used for creating the decoder_input_ids from the labels
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 64
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    # Helper method to to obtain inference for a given image
    def predict(self, image):
        # prepare image (i.e. resize + normalize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values_np = pixel_values.squeeze().numpy()
        # Predict
        self.model.eval()
        with torch.no_grad():
            pred = self.model.generate(pixel_values.to(self.device))
            string = self.processor.batch_decode(pred, skip_special_tokens=True)
        return string
