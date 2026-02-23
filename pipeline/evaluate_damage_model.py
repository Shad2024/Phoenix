import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

# ============================================
# PART 1: DATASET LOADER (unchanged)
# ============================================

class XView2Dataset:
    def __init__(self, root_dir):
        """
        root_dir: path to the main test folder containing 'images', 'labels', and 'targets'
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.targets_dir = os.path.join(root_dir, 'targets')
        
        # Check if directories exist
        for dir_path in [self.images_dir, self.labels_dir, self.targets_dir]:
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_path} not found!")
        
        # Build the file mapping
        self.data_pairs = self._build_file_mapping()
        print(f"Found {len(self.data_pairs)} unique disaster scenes")
    
    def _build_file_mapping(self):
        """Match pre/post images with their corresponding labels and targets"""
        
        # Get all image files
        all_images = glob.glob(os.path.join(self.images_dir, '*.png'))
        
        # Dictionary to store paired data
        pairs = {}
        
        for img_path in all_images:
            filename = os.path.basename(img_path)
            
            # Extract base ID from filename
            if '_pre_disaster' in filename:
                base_id = filename.replace('_pre_disaster.png', '')
                img_type = 'pre'
            elif '_post_disaster' in filename:
                base_id = filename.replace('_post_disaster.png', '')
                img_type = 'post'
            else:
                continue  # Skip unexpected files
            
            # Initialize dict for this base_id
            if base_id not in pairs:
                pairs[base_id] = {
                    'pre_image': None,
                    'post_image': None,
                    'label': None,
                    'pre_target': None,
                    'post_target': None
                }
            
            # Store image path
            pairs[base_id][f'{img_type}_image'] = img_path
            
            # Look for corresponding label file
            label_path = os.path.join(self.labels_dir, f'{base_id}.json')
            if os.path.exists(label_path):
                pairs[base_id]['label'] = label_path
            
            # Look for corresponding target masks
            pre_target = os.path.join(self.targets_dir, f'{base_id}_pre_disaster_target.png')
            post_target = os.path.join(self.targets_dir, f'{base_id}_post_disaster_target.png')
            
            if os.path.exists(pre_target):
                pairs[base_id]['pre_target'] = pre_target
            if os.path.exists(post_target):
                pairs[base_id]['post_target'] = post_target
        
        # Keep only entries with both pre and post images
        complete_pairs = {}
        for base_id, files in pairs.items():
            if files['pre_image'] and files['post_image']:
                complete_pairs[base_id] = files
        
        return complete_pairs
    
    def get_item(self, base_id):
        """Load all data for a specific base_id"""
        if base_id not in self.data_pairs:
            raise KeyError(f"Base ID {base_id} not found")
        
        files = self.data_pairs[base_id]
        
        # Load images
        pre_image = np.array(Image.open(files['pre_image']))
        post_image = np.array(Image.open(files['post_image']))
        
        data = {
            'base_id': base_id,
            'pre_image': pre_image,
            'post_image': post_image,
            'pre_image_path': files['pre_image'],
            'post_image_path': files['post_image']
        }
        
        # Load label JSON if exists
        if files['label'] and os.path.exists(files['label']):
            with open(files['label'], 'r') as f:
                data['label'] = json.load(f)
        
        # Load target masks if exist
        if files['pre_target'] and os.path.exists(files['pre_target']):
            data['pre_target'] = np.array(Image.open(files['pre_target']))
        if files['post_target'] and os.path.exists(files['post_target']):
            data['post_target'] = np.array(Image.open(files['post_target']))
        
        return data
    
    def get_all_base_ids(self):
        """Return all available base IDs"""
        return list(self.data_pairs.keys())
    
    def __len__(self):
        return len(self.data_pairs)


# ============================================
# PART 2: MODEL TESTER - COMPLETELY UPDATED
# ============================================

class XView2ModelTester:
    def __init__(self, model_path, dataset, device=None):
        """
        Initialize the tester with your model
        
        Args:
            model_path: Path to your downloaded model file
            dataset: XView2Dataset instance
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.dataset = dataset
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model - THIS WILL NOW WORK
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Damage class names for visualization
        self.class_names = {
            0: 'Background',
            1: 'No Damage',
            2: 'Minor Damage',
            3: 'Major Damage',
            4: 'Destroyed'
        }
        
        # Colors for visualization (RGB format)
        self.class_colors = {
            0: [0, 0, 0],        # Black
            1: [0, 255, 0],      # Green
            2: [255, 255, 0],    # Yellow
            3: [255, 165, 0],    # Orange
            4: [255, 0, 0]       # Red
        }
    
    def _load_model(self, model_path):
        """Load the SeresNext50 UNet model properly"""
        try:
            # First, load the state dict with weights_only=False since it's from a trusted source
            print("Loading state dict...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Extract the actual state dict based on common patterns
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("Found 'state_dict' key in checkpoint")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    print("Found 'model' key in checkpoint")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("Found 'model_state_dict' key in checkpoint")
                else:
                    # Assume the checkpoint itself is the state dict
                    state_dict = checkpoint
                    print("Using checkpoint directly as state dict")
            else:
                state_dict = checkpoint
                print("Checkpoint is not a dict, using as is")
            
            print("Creating SeresNext50 UNet model architecture...")
            
            # Install segmentation_models_pytorch if not available
            try:
                import segmentation_models_pytorch as smp
            except ImportError:
                print("Installing segmentation_models_pytorch...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'segmentation-models-pytorch'])
                import segmentation_models_pytorch as smp
                print("Installation complete!")
            
            # Create the model - UNet with seresnext50 encoder
            # Based on filename: seresnext50_unet_v2_512_fold3_fp16_crops
            model = smp.Unet(
                encoder_name='se_resnext50_32x4d',  # Full name for seresnext50
                encoder_weights=None,  # Don't load pretrained weights
                in_channels=6,  # 3 for pre + 3 for post = 6 channels
                classes=5,  # 5 damage classes (0-4)
            )
            
            # Clean state dict keys if needed (remove module. prefix if from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                else:
                    new_state_dict[k] = v
            
            # Load the state dict
            model.load_state_dict(new_state_dict, strict=False)
            print("State dict loaded successfully!")
            
            return model
            
        except Exception as e:
            print(f"Error in _load_model: {e}")
            print("\nTrying alternative approach with more details...")
            
            # Debug: Show what's in the checkpoint
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print(f"\nCheckpoint type: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint keys: {list(checkpoint.keys())}")
                    
                    # If there's a state dict, show its keys
                    if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                        state_dict_keys = list(checkpoint['state_dict'].keys())
                        print(f"State dict keys (first 5): {state_dict_keys[:5]}")
                        
                        # Try to infer input channels from first layer
                        first_key = state_dict_keys[0]
                        print(f"First layer shape: {checkpoint['state_dict'][first_key].shape}")
                else:
                    print(f"Checkpoint is not a dict, it's a {type(checkpoint)}")
            except:
                print("Could not inspect checkpoint further")
            
            raise
    
    def preprocess(self, image, target_size=(512, 512)):
        """Preprocess images for the UNet model (expects concatenated input)"""
        # Convert to float and normalize to [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Resize if needed
        if image.shape[:2] != target_size:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_image = pil_image.resize(target_size, Image.BILINEAR)
            image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        if len(image.shape) == 3:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            tensor = torch.from_numpy(image).unsqueeze(0).float()
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def predict(self, pre_image, post_image):
        """Run model prediction with concatenated input"""
        with torch.no_grad():
            # Ensure both are on the same device
            pre_image = pre_image.to(self.device)
            post_image = post_image.to(self.device)
            
            # Concatenate pre and post images along channel dimension
            # Both should be (C, H, W) at this point
            combined = torch.cat([pre_image, post_image], dim=0)  # -> (6, H, W)
            
            # Add batch dimension
            combined = combined.unsqueeze(0)  # -> (1, 6, H, W)
            
            # Run model
            output = self.model(combined)
            
            # Convert to class predictions
            if isinstance(output, tuple):
                output = output[0]
            
            if len(output.shape) == 4:  # (B, C, H, W)
                predictions = output.argmax(dim=1)
            else:
                predictions = output
            
            return predictions.cpu().numpy()
    
    def visualize_predictions(self, data, predictions, save_path=None):
        """Visualize model predictions alongside input images and ground truth"""
        
        # Create color visualization of predictions
        pred_viz = self._create_color_mask(predictions)
        
        plt.figure(figsize=(20, 10))
        
        # Pre-disaster image
        plt.subplot(2, 3, 1)
        plt.imshow(data['pre_image'])
        plt.title('Pre-Disaster')
        plt.axis('off')
        
        # Post-disaster image
        plt.subplot(2, 3, 2)
        plt.imshow(data['post_image'])
        plt.title('Post-Disaster')
        plt.axis('off')
        
        # Model predictions
        plt.subplot(2, 3, 3)
        plt.imshow(pred_viz)
        plt.title('Model Predictions')
        plt.axis('off')
        
        # Ground truth (if available)
        if 'post_target' in data:
            gt_viz = self._create_color_mask(data['post_target'])
            plt.subplot(2, 3, 4)
            plt.imshow(gt_viz)
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Difference map
            diff = (predictions != data['post_target']).astype(np.uint8) * 255
            plt.subplot(2, 3, 5)
            plt.imshow(diff, cmap='Reds')
            plt.title('Errors (Red = Mistake)')
            plt.axis('off')
            
            # Accuracy
            accuracy = np.mean(predictions == data['post_target'])
            plt.subplot(2, 3, 6)
            plt.text(0.1, 0.5, f'Pixel Accuracy: {accuracy:.2%}', 
                    fontsize=16, verticalalignment='center')
            plt.axis('off')
        
        plt.suptitle(f"Scene: {data['base_id']}", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _create_color_mask(self, mask):
        """Convert class indices to color image"""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in self.class_colors.items():
            color_mask[mask == class_idx] = color
        
        return color_mask
    
    def test_single_sample(self, base_id=None):
        """Test model on a single sample"""
        if base_id is None:
            base_ids = self.dataset.get_all_base_ids()
            if not base_ids:
                print("No samples found!")
                return
            base_id = base_ids[0]
        
        print(f"\n--- Testing on: {base_id} ---")
        
        # Load data
        data = self.dataset.get_item(base_id)
        
        # Preprocess
        pre_tensor = self.preprocess(data['pre_image'])
        post_tensor = self.preprocess(data['post_image'])
        
        # Predict
        predictions = self.predict(pre_tensor, post_tensor)
        predictions = predictions[0]  # Remove batch dimension
        
        # Resize predictions back to original size if needed
        if predictions.shape != data['pre_image'].shape[:2]:
            pil_pred = Image.fromarray(predictions.astype(np.uint8))
            pil_pred = pil_pred.resize((data['pre_image'].shape[1], 
                                        data['pre_image'].shape[0]), 
                                        Image.NEAREST)
            predictions = np.array(pil_pred)
        
        # Visualize
        self.visualize_predictions(data, predictions)
        
        # Print statistics
        unique, counts = np.unique(predictions, return_counts=True)
        print("\nPrediction distribution:")
        for u, c in zip(unique, counts):
            percentage = 100 * c / predictions.size
            print(f"  {self.class_names.get(u, f'Class {u}')}: {percentage:.1f}%")
        
        return predictions, data
    
    def evaluate_multiple_samples(self, num_samples=5):
        """Evaluate model on multiple samples"""
        base_ids = self.dataset.get_all_base_ids()[:num_samples]
        
        print(f"\n=== Evaluating on {len(base_ids)} samples ===\n")
        
        all_accuracies = []
        class_accuracies = defaultdict(list)
        
        for i, base_id in enumerate(base_ids, 1):
            print(f"[{i}/{len(base_ids)}] Processing: {base_id}")
            
            try:
                # Load data
                data = self.dataset.get_item(base_id)
                
                if 'post_target' not in data:
                    print(f"  No ground truth available, skipping...")
                    continue
                
                # Preprocess and predict
                pre_tensor = self.preprocess(data['pre_image'])
                post_tensor = self.preprocess(data['post_image'])
                predictions = self.predict(pre_tensor, post_tensor)[0]
                
                # Resize if needed
                if predictions.shape != data['post_target'].shape:
                    pil_pred = Image.fromarray(predictions.astype(np.uint8))
                    pil_pred = pil_pred.resize((data['post_target'].shape[1], 
                                                data['post_target'].shape[0]), 
                                                Image.NEAREST)
                    predictions = np.array(pil_pred)
                
                ground_truth = data['post_target']
                
                # Calculate overall accuracy
                accuracy = np.mean(predictions == ground_truth)
                all_accuracies.append(accuracy)
                
                # Calculate per-class accuracy
                for class_id in range(5):
                    class_mask = ground_truth == class_id
                    if np.sum(class_mask) > 0:
                        class_acc = np.mean(predictions[class_mask] == ground_truth[class_mask])
                        class_accuracies[class_id].append(class_acc)
                
                print(f"  Accuracy: {accuracy:.2%}")
                
            except Exception as e:
                print(f"  Error processing {base_id}: {e}")
        
        # Print summary
        self._print_evaluation_summary(all_accuracies, class_accuracies)
        
        return all_accuracies, class_accuracies
    
    def _print_evaluation_summary(self, accuracies, class_accuracies):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"Overall Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
        
        print("\nPer-Class Accuracy:")
        for class_id in sorted(class_accuracies.keys()):
            if class_accuracies[class_id]:
                mean_class_acc = np.mean(class_accuracies[class_id])
                std_class_acc = np.std(class_accuracies[class_id])
                print(f"  {self.class_names[class_id]}: {mean_class_acc:.2%} ± {std_class_acc:.2%}")
    
    def batch_predict(self, output_dir='predictions'):
        """Run prediction on all samples and save results"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_ids = self.dataset.get_all_base_ids()
        print(f"\nRunning batch prediction on {len(base_ids)} samples...")
        
        for i, base_id in enumerate(base_ids):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(base_ids)}")
            
            try:
                # Load data
                data = self.dataset.get_item(base_id)
                
                # Predict
                pre_tensor = self.preprocess(data['pre_image'])
                post_tensor = self.preprocess(data['post_image'])
                predictions = self.predict(pre_tensor, post_tensor)[0]
                
                # Resize to original image size
                if predictions.shape != data['pre_image'].shape[:2]:
                    pil_pred = Image.fromarray(predictions.astype(np.uint8))
                    pil_pred = pil_pred.resize((data['pre_image'].shape[1], 
                                                data['pre_image'].shape[0]), 
                                                Image.NEAREST)
                    predictions = np.array(pil_pred)
                
                # Save prediction
                pred_path = os.path.join(output_dir, f"{base_id}_prediction.png")
                Image.fromarray(predictions.astype(np.uint8)).save(pred_path)
                
                # Also save color visualization
                color_viz = self._create_color_mask(predictions)
                viz_path = os.path.join(output_dir, f"{base_id}_color.png")
                Image.fromarray(color_viz).save(viz_path)
                
            except Exception as e:
                print(f"  Error on {base_id}: {e}")
        
        print(f"\nComplete! Predictions saved to {output_dir}/")


# ============================================
# PART 3: MAIN EXECUTION
# ============================================

def main():
    """Main function to run everything"""
    
    # ===== CONFIGURATION - CHANGE THESE PATHS =====
    DATA_ROOT = r"C:\Users\bolaky\Downloads\test_images_labels_targets\test" 
    MODEL_PATH = r"C:\Users\bolaky\PycharmProjects\Rebuild\Models\Dec21_11_50_seresnext50_unet_v2_512_fold3_fp16_crops.pth"
    # ==============================================
    
    print("="*60)
    print("xView2 Model Testing Script")
    print("="*60)
    
    # Step 1: Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = XView2Dataset(DATA_ROOT)
    
    if len(dataset) == 0:
        print("❌ No valid scenes found! Please check your DATA_ROOT path.")
        return
    
    print(f"✅ Found {len(dataset)} scenes")
    
    # Step 2: Initialize model tester
    print("\n[2/4] Initializing model tester...")
    try:
        tester = XView2ModelTester(MODEL_PATH, dataset)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nPlease check:")
        print("1. Model path is correct")
        print("2. Model file is not corrupted")
        print("3. You may need to adapt _load_model() for your specific model")
        return
    
    # Step 3: Test on a single sample
    print("\n[3/4] Testing on a single sample...")
    base_ids = dataset.get_all_base_ids()
    print(f"Sample IDs available: {base_ids[:3]}...")
    
    try:
        predictions, data = tester.test_single_sample(base_ids[0])
    except Exception as e:
        print(f"❌ Single sample test failed: {e}")
        print("\nThis might be due to:")
        print("1. Model expects different input format (adapt preprocess())")
        print("2. Model requires specific preprocessing")
        print("3. Image size mismatch")
        return
    
    # Step 4: Evaluate on multiple samples (optional)
    print("\n[4/4] Would you like to evaluate on multiple samples?")
    response = input("Run evaluation on 5 samples? (y/n): ").lower()
    
    if response == 'y':
        try:
            accuracies, class_accs = tester.evaluate_multiple_samples(num_samples=5)
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    print("\n✅ Testing complete!")
    print("\nNext steps:")
    print("1. If predictions look good, run batch_predict() on all images")
    print("2. If there were errors, adapt the model-specific functions")
    print("3. Check the 'predictions/' folder for output images")


if __name__ == "__main__":
    main()