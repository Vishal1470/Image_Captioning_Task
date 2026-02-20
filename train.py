from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    print("Model loaded successfully!")

if __name__ == "__main__":
    main()
