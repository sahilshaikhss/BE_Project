import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from PIL import ImageFile

# ✅ Fix truncated image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

class EffNetCBIR:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.feature_extractor = self._build_feature_extractor()
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')

    def _build_feature_extractor(self):
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        base_model.trainable = False

        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        features = layers.Dense(256, activation='relu', name='feature_vector')(x)

        return models.Model(inputs=inputs, outputs=features)

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=self.input_shape[:2])
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def extract_features(self, img_path):
        try:
            img = self.preprocess_image(img_path)
            features = self.feature_extractor.predict(img, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"❌ Failed to extract features from {img_path}: {e}")
            return None

    def build_image_database(self, image_folder):
        self.image_paths = []
        self.features = []

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        print(f"\n📁 Building image database from {image_folder}")
        for root, _, files in os.walk(image_folder):
            for file in tqdm(files):
                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, file)
                    features = self.extract_features(img_path)
                    if features is not None:
                        self.image_paths.append(img_path)
                        self.features.append(features)

        self.features = np.array(self.features)
        self.image_paths = np.array(self.image_paths)

        if len(self.features) == 0:
            raise RuntimeError("❌ No valid images found. Cannot build the database.")

        self.knn.fit(self.features)
        print(f"✅ Database built with {len(self.image_paths)} images.")

    def query_image(self, query_img_path, num_results=5):
        query_features = self.extract_features(query_img_path)
        if query_features is None:
            raise ValueError(f"❌ Could not extract features from query image: {query_img_path}")

        num_available = len(self.image_paths)
        k = min(num_results, num_available)

        distances, indices = self.knn.kneighbors(
            query_features.reshape(1, -1),
            n_neighbors=k
        )

        similar_paths = self.image_paths[indices[0]]
        distances = distances[0]

        return similar_paths, distances

    def visualize_results(self, query_path, result_paths, distances):
        plt.figure(figsize=(15, 8))

        # Query image
        plt.subplot(1, len(result_paths) + 1, 1)
        query_img = mpimg.imread(query_path)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis('off')

        # Similar images
        for i, (path, dist) in enumerate(zip(result_paths, distances)):
            plt.subplot(1, len(result_paths) + 1, i + 2)
            img = mpimg.imread(path)
            plt.imshow(img)
            plt.title(f"Score: {1 - dist:.3f}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
