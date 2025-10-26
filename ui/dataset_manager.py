"""
Image Upload and Dataset Management Module
Handles image uploads, dataset browsing, and data management through the UI
"""

import streamlit as st
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import zipfile
import tempfile
from datetime import datetime
import json
import cv2
from typing import List, Dict, Optional, Tuple

class DatasetManager:
    """Manages datasets and image uploads for the analysis pipeline"""
    
    def __init__(self, upload_dir="./data/uploads", processed_dir="./data/processed"):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session state for dataset management
        if 'uploaded_images' not in st.session_state:
            st.session_state.uploaded_images = []
        if 'dataset_metadata' not in st.session_state:
            st.session_state.dataset_metadata = {}
    
    def render_dataset_management_panel(self):
        """Render the complete dataset management interface"""
        st.header("📁 Dataset Management")
        
        # Dataset source selection
        dataset_source = st.selectbox(
            "Choose dataset source:",
            [
                "Hugging Face Flickr30k", 
                "Upload Images", 
                "Upload ZIP Archive",
                "Local Directory",
                "Existing Dataset"
            ]
        )
        
        if dataset_source == "Upload Images":
            self._render_image_upload_interface()
        elif dataset_source == "Upload ZIP Archive":
            self._render_zip_upload_interface()
        elif dataset_source == "Local Directory":
            self._render_local_directory_interface()
        elif dataset_source == "Existing Dataset":
            self._render_existing_dataset_interface()
        else:  # Hugging Face Flickr30k
            self._render_huggingface_interface()
        
        # Dataset browser
        st.divider()
        self._render_dataset_browser()
        
        return dataset_source
    
    def _render_image_upload_interface(self):
        """Interface for uploading individual images"""
        st.subheader("📤 Upload Individual Images")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            help="Upload one or more image files"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**{len(uploaded_files)} files selected**")
                
                # Progress bar for upload
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process uploads
                if st.button("📥 Process Uploads", type="primary"):
                    processed_files = []
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Save file
                        file_path = self._save_uploaded_file(uploaded_file)
                        if file_path:
                            processed_files.append({
                                'filename': uploaded_file.name,
                                'path': str(file_path),
                                'size': uploaded_file.size,
                                'upload_time': datetime.now().isoformat()
                            })
                    
                    # Update session state
                    st.session_state.uploaded_images.extend(processed_files)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Upload complete!")
                    st.success(f"Successfully uploaded {len(processed_files)} images!")
            
            with col2:
                # Show file preview
                if uploaded_files:
                    st.write("**Preview:**")
                    preview_file = uploaded_files[0]  # Show first file
                    
                    try:
                        image = Image.open(preview_file)
                        st.image(image, caption=preview_file.name, use_column_width=True)
                        
                        # Show file info
                        st.write(f"**Size:** {preview_file.size:,} bytes")
                        st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]}")
                        st.write(f"**Format:** {image.format}")
                    except Exception as e:
                        st.error(f"Error loading preview: {e}")
    
    def _render_zip_upload_interface(self):
        """Interface for uploading ZIP archives containing images"""
        st.subheader("📦 Upload ZIP Archive")
        
        uploaded_zip = st.file_uploader(
            "Choose ZIP file",
            type=['zip'],
            help="Upload a ZIP file containing images"
        )
        
        if uploaded_zip:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Archive:** {uploaded_zip.name}")
                st.write(f"**Size:** {uploaded_zip.size:,} bytes")
                
                # Extract and process ZIP
                if st.button("📂 Extract and Process", type="primary"):
                    with st.spinner("Extracting archive..."):
                        extracted_files = self._process_zip_upload(uploaded_zip)
                        
                        if extracted_files:
                            st.success(f"✅ Extracted {len(extracted_files)} images from archive!")
                            st.session_state.uploaded_images.extend(extracted_files)
                        else:
                            st.warning("No valid images found in archive")
            
            with col2:
                # Show archive contents preview
                try:
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_file:
                        file_list = zip_file.namelist()
                        image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in self.supported_formats)]
                        
                        st.write(f"**Total files:** {len(file_list)}")
                        st.write(f"**Image files:** {len(image_files)}")
                        
                        if image_files:
                            st.write("**Sample images:**")
                            for img_file in image_files[:5]:  # Show first 5
                                st.write(f"• {img_file}")
                            if len(image_files) > 5:
                                st.write(f"... and {len(image_files) - 5} more")
                
                except Exception as e:
                    st.error(f"Error reading archive: {e}")
    
    def _render_local_directory_interface(self):
        """Interface for selecting local directory"""
        st.subheader("📁 Local Directory")
        
        directory_path = st.text_input(
            "Directory path:",
            value="./data/images",
            help="Path to directory containing images"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("📂 Scan Directory", type="primary"):
                if os.path.exists(directory_path):
                    image_files = self._scan_directory(directory_path)
                    
                    if image_files:
                        st.session_state.uploaded_images = image_files
                        st.success(f"✅ Found {len(image_files)} images in directory!")
                    else:
                        st.warning("No valid images found in directory")
                else:
                    st.error("Directory does not exist")
        
        with col2:
            if os.path.exists(directory_path):
                try:
                    files = os.listdir(directory_path)
                    image_files = [f for f in files if any(f.lower().endswith(ext) for ext in self.supported_formats)]
                    
                    st.write(f"**Total files:** {len(files)}")
                    st.write(f"**Image files:** {len(image_files)}")
                    
                    if image_files:
                        st.write("**Sample files:**")
                        for img_file in image_files[:5]:
                            st.write(f"• {img_file}")
                        if len(image_files) > 5:
                            st.write(f"... and {len(image_files) - 5} more")
                
                except Exception as e:
                    st.error(f"Error scanning directory: {e}")
    
    def _render_existing_dataset_interface(self):
        """Interface for loading existing processed datasets"""
        st.subheader("💾 Existing Datasets")
        
        # Scan for existing datasets
        existing_datasets = self._find_existing_datasets()
        
        if existing_datasets:
            dataset_name = st.selectbox(
                "Select dataset:",
                options=list(existing_datasets.keys()),
                help="Choose from previously processed datasets"
            )
            
            if dataset_name:
                dataset_info = existing_datasets[dataset_name]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Images:** {dataset_info.get('image_count', 'Unknown')}")
                    st.write(f"**Created:** {dataset_info.get('created_date', 'Unknown')}")
                    st.write(f"**Size:** {dataset_info.get('total_size', 'Unknown')}")
                
                with col2:
                    if st.button("📥 Load Dataset", type="primary"):
                        loaded_images = self._load_existing_dataset(dataset_name)
                        if loaded_images:
                            st.session_state.uploaded_images = loaded_images
                            st.success(f"✅ Loaded {len(loaded_images)} images!")
        else:
            st.info("No existing datasets found")
    
    def _render_huggingface_interface(self):
        """Interface for Hugging Face dataset configuration"""
        st.subheader("🤗 Hugging Face Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "Dataset name:",
                value="nlphuji/flickr30k",
                help="Hugging Face dataset identifier"
            )
            
            subset = st.text_input(
                "Subset (optional):",
                value="",
                help="Dataset subset if applicable"
            )
            
            split = st.selectbox(
                "Split:",
                ["train", "test", "validation", "all"],
                index=0,
                help="Dataset split to use"
            )
        
        with col2:
            sample_size = st.number_input(
                "Sample size:",
                min_value=100,
                max_value=10000,
                value=2000,
                help="Number of images to download"
            )
            
            filter_captions = st.checkbox(
                "Filter by captions",
                value=True,
                help="Only include images with people-related captions"
            )
            
            cache_locally = st.checkbox(
                "Cache locally",
                value=True,
                help="Save downloaded images locally"
            )
        
        if st.button("📥 Download Dataset", type="primary"):
            with st.spinner("Downloading dataset..."):
                # This would integrate with the actual HuggingFace collector
                st.info("Dataset download initiated. This will be handled by the pipeline.")
    
    def _render_dataset_browser(self):
        """Render dataset browser and management interface"""
        st.subheader("🔍 Dataset Browser")
        
        if not st.session_state.uploaded_images:
            st.info("No images loaded. Please upload or select a dataset above.")
            return
        
        # Dataset statistics
        total_images = len(st.session_state.uploaded_images)
        total_size = sum(img.get('size', 0) for img in st.session_state.uploaded_images)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", f"{total_images:,}")
        with col2:
            st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
        with col3:
            avg_size = total_size / total_images if total_images > 0 else 0
            st.metric("Avg Size", f"{avg_size / 1024:.1f} KB")
        with col4:
            st.metric("Status", "Ready" if total_images > 0 else "Empty")
        
        # Image grid browser
        st.write("**Image Gallery:**")
        
        # Pagination controls
        images_per_page = st.selectbox("Images per page:", [12, 24, 48], index=1)
        
        total_pages = (total_images + images_per_page - 1) // images_per_page
        
        if total_pages > 1:
            page = st.number_input(
                "Page:",
                min_value=1,
                max_value=total_pages,
                value=1
            ) - 1  # Convert to 0-based index
        else:
            page = 0
        
        # Display images in grid
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)
        page_images = st.session_state.uploaded_images[start_idx:end_idx]
        
        # Create grid layout
        cols_per_row = 4
        rows = (len(page_images) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                
                if img_idx < len(page_images):
                    image_info = page_images[img_idx]
                    
                    with cols[col_idx]:
                        try:
                            # Load and display image
                            image_path = image_info['path']
                            if os.path.exists(image_path):
                                image = Image.open(image_path)
                                st.image(
                                    image,
                                    caption=image_info['filename'],
                                    use_column_width=True
                                )
                                
                                # Image actions
                                if st.button(f"🗑️", key=f"delete_{start_idx + img_idx}"):
                                    self._delete_image(start_idx + img_idx)
                                    st.rerun()
                            else:
                                st.error(f"Image not found: {image_info['filename']}")
                        
                        except Exception as e:
                            st.error(f"Error loading {image_info['filename']}: {e}")
        
        # Dataset actions
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Dataset"):
                saved_path = self._save_dataset()
                if saved_path:
                    st.success(f"Dataset saved to {saved_path}")
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with col3:
            if st.button("📊 Analyze Images"):
                self._analyze_dataset_properties()
        
        with col4:
            if st.button("🗑️ Clear All"):
                if st.confirm("Are you sure you want to clear all images?"):
                    st.session_state.uploaded_images = []
                    st.rerun()
    
    def _save_uploaded_file(self, uploaded_file) -> Optional[Path]:
        """Save an uploaded file to the upload directory"""
        try:
            # Create unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{uploaded_file.name}"
            file_path = self.upload_dir / filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return file_path
        
        except Exception as e:
            st.error(f"Error saving file {uploaded_file.name}: {e}")
            return None
    
    def _process_zip_upload(self, uploaded_zip) -> List[Dict]:
        """Process uploaded ZIP file and extract images"""
        extracted_files = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP to temporary directory
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
                
                # Find and process image files
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in self.supported_formats):
                            src_path = os.path.join(root, file)
                            
                            # Create unique filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            unique_filename = f"{timestamp}_{file}"
                            dst_path = self.upload_dir / unique_filename
                            
                            # Copy file to upload directory
                            shutil.copy2(src_path, dst_path)
                            
                            # Get file info
                            file_size = os.path.getsize(dst_path)
                            
                            extracted_files.append({
                                'filename': file,
                                'path': str(dst_path),
                                'size': file_size,
                                'upload_time': datetime.now().isoformat(),
                                'source': 'zip_archive'
                            })
        
        except Exception as e:
            st.error(f"Error processing ZIP file: {e}")
        
        return extracted_files
    
    def _scan_directory(self, directory_path: str) -> List[Dict]:
        """Scan directory for image files"""
        image_files = []
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.supported_formats):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        
                        image_files.append({
                            'filename': file,
                            'path': file_path,
                            'size': file_size,
                            'upload_time': datetime.now().isoformat(),
                            'source': 'local_directory'
                        })
        
        except Exception as e:
            st.error(f"Error scanning directory: {e}")
        
        return image_files
    
    def _find_existing_datasets(self) -> Dict:
        """Find existing processed datasets"""
        datasets = {}
        
        try:
            if self.processed_dir.exists():
                for dataset_dir in self.processed_dir.iterdir():
                    if dataset_dir.is_dir():
                        metadata_file = dataset_dir / "metadata.json"
                        
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            datasets[dataset_dir.name] = metadata
                        else:
                            # Create basic metadata for directories without it
                            image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
                            datasets[dataset_dir.name] = {
                                'image_count': len(image_files),
                                'created_date': 'Unknown',
                                'total_size': 'Unknown'
                            }
        
        except Exception as e:
            st.error(f"Error finding existing datasets: {e}")
        
        return datasets
    
    def _load_existing_dataset(self, dataset_name: str) -> List[Dict]:
        """Load an existing dataset"""
        dataset_path = self.processed_dir / dataset_name
        image_files = []
        
        try:
            for file_path in dataset_path.iterdir():
                if file_path.suffix.lower() in self.supported_formats:
                    file_size = file_path.stat().st_size
                    
                    image_files.append({
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size': file_size,
                        'upload_time': datetime.now().isoformat(),
                        'source': 'existing_dataset'
                    })
        
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
        
        return image_files
    
    def _delete_image(self, index: int):
        """Delete an image from the dataset"""
        if 0 <= index < len(st.session_state.uploaded_images):
            image_info = st.session_state.uploaded_images[index]
            
            # Delete file if it exists
            if os.path.exists(image_info['path']):
                try:
                    os.remove(image_info['path'])
                except Exception as e:
                    st.error(f"Error deleting file: {e}")
            
            # Remove from session state
            st.session_state.uploaded_images.pop(index)
    
    def _save_dataset(self) -> Optional[str]:
        """Save current dataset with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"dataset_{timestamp}"
            dataset_path = self.processed_dir / dataset_name
            dataset_path.mkdir(exist_ok=True)
            
            # Copy images to dataset directory
            for img_info in st.session_state.uploaded_images:
                src_path = Path(img_info['path'])
                dst_path = dataset_path / src_path.name
                shutil.copy2(src_path, dst_path)
            
            # Save metadata
            metadata = {
                'dataset_name': dataset_name,
                'created_date': datetime.now().isoformat(),
                'image_count': len(st.session_state.uploaded_images),
                'total_size': sum(img.get('size', 0) for img in st.session_state.uploaded_images),
                'images': st.session_state.uploaded_images
            }
            
            metadata_path = dataset_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(dataset_path)
        
        except Exception as e:
            st.error(f"Error saving dataset: {e}")
            return None
    
    def _analyze_dataset_properties(self):
        """Analyze and display dataset properties"""
        if not st.session_state.uploaded_images:
            st.warning("No images to analyze")
            return
        
        st.subheader("📊 Dataset Analysis")
        
        with st.spinner("Analyzing images..."):
            # Analyze image properties
            widths, heights, formats, sizes = [], [], [], []
            
            for img_info in st.session_state.uploaded_images[:50]:  # Limit to first 50 for performance
                try:
                    with Image.open(img_info['path']) as img:
                        widths.append(img.width)
                        heights.append(img.height)
                        formats.append(img.format)
                        sizes.append(img_info.get('size', 0))
                except Exception:
                    continue
            
            if widths:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dimension Statistics:**")
                    st.write(f"Width: {min(widths)} - {max(widths)} (avg: {np.mean(widths):.0f})")
                    st.write(f"Height: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.0f})")
                    st.write(f"Aspect ratios: {min(np.array(widths)/np.array(heights)):.2f} - {max(np.array(widths)/np.array(heights)):.2f}")
                
                with col2:
                    st.write("**Format Distribution:**")
                    format_counts = pd.Series(formats).value_counts()
                    for fmt, count in format_counts.items():
                        st.write(f"{fmt}: {count} images")