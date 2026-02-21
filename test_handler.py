import handler
import base64
import os
from PIL import Image
from io import BytesIO

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_handler():
    # Mock job data
    # Ensure you have a test image at 'examples/B.jpg' or change the path
    test_image_path = "examples/B.jpg"
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Please provide a valid image path.")
        return

    test_image_base64 = image_to_base64(test_image_path)

    job = {
        "id": "test_job_123",
        "input": {
            "images": [
                {"image_base64": test_image_base64}
            ],
            "prompt": "transform into anime",
            "lora_adapter": "Photo-to-Anime",
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "seed": 42
        }
    }

    print("🚀 Starting local test...")
    try:
        result = handler.handler(job)
        
        if "error" in result:
            print(f"❌ Test failed with error: {result['error']}")
        else:
            print("✅ Test successful!")
            for idx, img_data in enumerate(result.get("images", [])):
                if "image_url" in img_data:
                    print(f"Image {idx} URL: {img_data['image_url']}")
                if "image_base64" in img_data:
                    print(f"Image {idx} Base64 (first 50 chars): {img_data['image_base64'][:50]}...")
                    
                    # Save local copy for verification
                    img_bytes = base64.b64decode(img_data['image_base64'])
                    img = Image.open(BytesIO(img_bytes))
                    img.save(f"test_output_{idx}.png")
                    print(f"Saved output to test_output_{idx}.png")

    except Exception as e:
        print(f"❌ An error occurred during testing: {e}")

if __name__ == "__main__":
    # Set environment variables for local testing if needed
    # os.environ["R2_ACCOUNT_ID"] = "..."
    # os.environ["R2_ACCESS_KEY_ID"] = "..."
    # os.environ["R2_SECRET_ACCESS_KEY"] = "..."
    # os.environ["R2_BUCKET_NAME"] = "..."
    # os.environ["R2_PUBLIC_URL"] = "..."
    
    test_handler()
