import insightface
import numpy as np
import cv2
import logging
import os

# Configure logging for insightface (optional but helpful)
# logging.basicConfig(level=logging.INFO)

class FaceRecognizer:
    """
    A wrapper class for using insightface models to extract face embeddings (ID features).
    """
    def __init__(self, model_name='buffalo_l', providers=None):
        """
        Initializes the FaceRecognizer.

        Args:
            model_name (str): The name of the face recognition model pack to use
                              (e.g., 'buffalo_l', 'antelopev2'). 'buffalo_l' is a
                              good general-purpose choice.
            providers (list, optional): List of ONNX Runtime execution providers
                                       (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
                                       If None, insightface will attempt to use CUDA if available,
                                       otherwise CPU.
        """
        self.model_name = model_name
        self.providers = providers if providers else ['CUDAExecutionProvider', 'CPUExecutionProvider']

        try:
            # Load the FaceAnalysis app which includes detection, alignment, and recognition
            # detection model ('retinaface_r50_v1' or 'scrfd_10g_bnkps' are common)
            # recognition model is implicitly loaded by 'buffalo_l' or other specified name
            self.app = insightface.app.FaceAnalysis(name=model_name,
                                                    allowed_modules=['detection', 'recognition'], # Ensure recognition is loaded
                                                    providers=self.providers)
            # Prepare the model, setting detection thresholds etc.
            # You might adjust ctx_id (GPU device id, -1 for CPU) and thresholds
            self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
            logging.info(f"Insightface FaceAnalysis app initialized successfully with model '{model_name}'.")
            # Perform a dummy inference to potentially speed up the first real call
            try:
                _ = self.app.get(np.zeros((128, 128, 3), dtype=np.uint8))
                logging.info("Performed dummy inference.")
            except Exception as e:
                 logging.warning(f"Dummy inference failed (this might be okay): {e}")


        except Exception as e:
            logging.error(f"Failed to initialize Insightface FaceAnalysis app: {e}")
            logging.error("Please ensure insightface, onnxruntime, and model files are installed correctly.")
            logging.error("Try installing CPU provider if CUDA is unavailable: pip install onnxruntime")
            logging.error("Or GPU provider: pip install onnxruntime-gpu")
            self.app = None
            raise RuntimeError(f"FaceRecognizer initialization failed: {e}")

    def get_embedding(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """
        Extracts the face embedding (ID feature) from a single face in an image.

        Args:
            image_bgr (np.ndarray): The input image in BGR format (as loaded by cv2.imread).

        Returns:
            np.ndarray | None: A 1D NumPy array representing the 512-dimensional face embedding,
                               or None if no face is detected or an error occurs.
        """
        if self.app is None:
            logging.error("FaceAnalysis app is not initialized.")
            return None

        try:
            # Use the app to get face information (includes detection and embedding extraction)
            faces = self.app.get(image_bgr)

            if not faces:
                # logging.warning("No face detected in the image.")
                return None

            # --- Assumption: We only care about the first detected face ---
            # In face swapping, usually, there's one main target face.
            # If multiple faces are a concern, you might need logic to select the largest
            # or center-most face.
            first_face = faces[0]

            # Extract the embedding (feature vector)
            embedding = first_face.get('embedding')
            if embedding is None:
                logging.warning("Face detected, but embedding could not be extracted.")
                return None

            # Normalize the embedding (insightface embeddings are typically already normalized,
            # but doing it again ensures consistency)
            norm_embedding = embedding / np.linalg.norm(embedding)

            return norm_embedding # Should be a (512,) shape NumPy array

        except Exception as e:
            logging.error(f"Error during face embedding extraction: {e}")
            return None

# --- Testing Block ---
if __name__ == "__main__":
    print("--- Testing FaceRecognizer ---")

    # --- Configuration ---
    # Create a dummy image file path for testing
    # In a real scenario, replace this with a path to an actual image with a face
    # Make sure the test image exists!
    # Let's create a dummy black image for basic testing if no image is provided
    test_image_path = "/root/HiFiVFS/data/test_image.jpg" # Change this to your test image path
    if not os.path.exists(test_image_path):
         print(f"Creating a dummy black image for testing at: {test_image_path}")
         dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
         cv2.imwrite(test_image_path, dummy_img)
         # Note: Detection will likely fail on a black image, testing primarily the loading mechanism.
         # For real testing, provide an actual face image path here.
         # test_image_path = "path/to/your/test/image_with_face.jpg"


    # --- Initialization ---
    try:
        print("Initializing FaceRecognizer...")
        # Initialize with default settings (will try CUDA first, then CPU)
        recognizer = FaceRecognizer(model_name='buffalo_l')
        print("FaceRecognizer initialized.")
    except Exception as e:
        print(f"ERROR: Could not initialize FaceRecognizer: {e}")
        recognizer = None

    # --- Feature Extraction ---
    if recognizer and os.path.exists(test_image_path):
        print(f"\nLoading test image: {test_image_path}")
        try:
            # Load the image using OpenCV
            img_bgr = cv2.imread(test_image_path)

            if img_bgr is None:
                print(f"ERROR: Failed to load image {test_image_path}")
            else:
                print("Image loaded successfully.")
                print("Extracting embedding...")
                # Get the embedding
                embedding = recognizer.get_embedding(img_bgr)

                if embedding is not None:
                    print("\n--- Embedding Extraction Successful ---")
                    print(f"Embedding Type: {type(embedding)}")
                    print(f"Embedding Shape: {embedding.shape}") # Should be (512,)
                    print(f"Embedding Norm (should be close to 1): {np.linalg.norm(embedding):.4f}")
                    # print(f"Embedding Snippet: {embedding[:10]}...") # Print first 10 values
                    # --- 打印具体的数值 ---
                    # print(f"Embedding Snippet: {embedding[:10]}...") # 打印前 10 个值
                    print("\nFull Embedding Vector:")
                    print(embedding) # 打印完整的 512 个值
                else:
                    print("\n--- Embedding Extraction Failed ---")
                    print("Could not extract embedding. Check if a face is clearly visible in the test image or if there were errors during initialization.")

        except Exception as e:
            print(f"ERROR during embedding extraction test: {e}")
    elif not recognizer:
         print("Skipping embedding extraction test as recognizer failed to initialize.")
    elif not os.path.exists(test_image_path):
        print(f"Skipping embedding extraction test as test image '{test_image_path}' does not exist.")

    print("\n--- Test Finished ---")