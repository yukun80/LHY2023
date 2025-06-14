# Professor Hung-yi Lee's ML2023 HW2: Complete Analysis and Optimization Guide

Professor Hung-yi Lee's Machine Learning 2023 Spring HW2 presents a **phoneme classification challenge** that serves as a crucial stepping stone in understanding speech recognition and deep neural networks. This assignment transforms raw audio signals into discrete phoneme labels, requiring students to master both signal processing fundamentals and advanced neural architectures.

## The phoneme classification challenge

HW2 tasks students with **framewise phoneme classification using deep neural networks** on speech data derived from the TIMIT corpus. Students must classify audio frames into one of 41 distinct phoneme categories, progressing through increasingly sophisticated baselines: Simple (49.8% accuracy), Medium (66.4%), Strong (74.9%), and the coveted Boss baseline (83.0% accuracy). The assignment operates as a Kaggle competition with strict submission limits of 5 per day, emphasizing thoughtful experimentation over trial-and-error approaches.

The dataset consists of continuous speech recordings segmented into frames, with each frame represented by acoustic features (typically MFCCs) and labeled with corresponding phoneme categories. This framewise approach mirrors real-world automatic speech recognition systems, making the assignment highly relevant to industry applications.

## Base code structure and progression pathway

The provided framework follows a **systematic progression philosophy** that guides students from basic neural networks to state-of-the-art architectures. The base code includes essential components: data preprocessing pipelines for acoustic feature extraction, model architecture definitions with TODO sections for improvements, training loops with proper validation procedures, TensorBoard integration for monitoring, and learning rate scheduling capabilities.

**Architecture Evolution Path:**
- **Models 1-4**: Simple feedforward networks establishing baseline functionality
- **Models 5-8**: Deeper networks with 2-3 hidden layers and regularization
- **Models 9-12**: Advanced feedforward approaches with sophisticated optimization
- **Models 13-16**: RNN/LSTM architectures capturing temporal dependencies
- **Boss baseline**: Bidirectional LSTM networks with attention mechanisms and advanced optimization

The code structure emphasizes modularity, allowing students to experiment with different components while maintaining consistent evaluation procedures. PyTorch serves as the primary framework, with extensive use of TensorBoard for training visualization and hyperparameter tracking.

## Advanced optimization strategies for maximum performance

Achieving the Boss baseline requires **systematic application of multiple optimization techniques** working in harmony. The most impactful architectural choice involves implementing Bidirectional LSTM networks, which achieved a record 17.7% test error on TIMIT benchmarks by leveraging contextual information from both past and future frames.

**Feature Engineering Excellence:**
MFCC extraction should use 25ms windows with 10ms shifts, applying pre-emphasis filtering (α ≈ 0.97) and 40 mel-scale filter banks. The first 12-13 coefficients provide optimal information density while maintaining computational efficiency. Advanced practitioners implement spectrotemporal modulation features and multi-resolution analysis for phoneme group-specific optimization.

**Training Optimization Protocol:**
Learning rate represents the most critical hyperparameter, with systematic exploration starting at 1e-3 and ranging [1e-4, 1e-2]. Implement cosine annealing or exponential decay schedules for stable convergence. Batch sizes of 32-128 work effectively, with batch normalization and dropout (0.1-0.5) providing essential regularization. L2 weight decay around 1e-4 prevents overfitting without constraining model capacity.

**Advanced Architectural Techniques:**
Successful students implement hierarchical phoneme clustering, recognizing that similar phonemes (fricatives, vowels, stops) benefit from specialized sub-models. Attention mechanisms enable direct learning of global temporal dependencies, while ensemble methods combining multiple architectural approaches consistently improve robustness. Context-invariant representation learning through multi-task objectives provides additional performance gains.

## Systematic improvement methodology for beginners

The key to success lies in **following a disciplined OODA Loop approach**: Observe training metrics and confusion matrices, Orient by analyzing error patterns and identifying bottlenecks, Decide on targeted improvements based on impact analysis, and Act with systematic changes while documenting results.

**Phase 1 Foundation (Week 1-2):**
Establish a minimal viable implementation achieving >40% accuracy on a 5-class subset. Focus exclusively on getting the basic pipeline working: data loading, simple CNN/MLP architecture, training loop, and evaluation metrics. This foundation phase is crucial—resist the temptation to add complexity before achieving basic functionality.

**Phase 2 Data-Centric Improvements (Week 3-4):**
Expand dataset coverage and implement robust preprocessing. Audio quality enhancement through noise reduction, amplitude normalization, and silence trimming provides immediate improvements. Feature extraction optimization with proper MFCC parameters and context windowing typically yields 60-70% accuracy.

**Phase 3 Model Optimization (Week 5-6):**
Transition to temporal modeling with LSTM/BiLSTM architectures. Systematic hyperparameter tuning using random search followed by grid search in promising regions. Implement comprehensive regularization strategies. Target 80-85% accuracy through architectural sophistication.

**Phase 4 Advanced Techniques (Week 7-8):**
Deploy ensemble methods, transfer learning from pre-trained speech models, and advanced data augmentation strategies. Focus on the final performance push toward 90%+ accuracy through technique combination and fine-tuning.

## Common pitfalls and debugging strategies

**Critical Mistake 1: Inadequate Data Preprocessing**
Students frequently underestimate preprocessing importance, leading to training convergence without generalization improvement. Implement systematic audio quality checks, consistent format conversion, and proper feature normalization. Use validation curves to detect preprocessing issues early.

**Critical Mistake 2: Architecture-Data Mismatch**
Many students apply image-focused architectures to temporal speech data without considering sequential dependencies. Speech requires temporal modeling—transition from CNNs to RNNs/LSTMs as soon as basic functionality is achieved.

**Critical Mistake 3: Hyperparameter Neglect**
Learning rate selection can make the difference between 60% and 80% accuracy. Implement learning rate finder techniques, monitor gradient flow, and use adaptive scheduling. Document all hyperparameter experiments for systematic optimization.

**Debugging Protocol:**
When models fail to learn, verify single-batch overfitting capability first. Check gradient flow and ensure reasonable hyperparameters. When overfitting occurs, implement regularization progressively: dropout, batch normalization, weight decay, then data augmentation. When performance plateaus, analyze confusion matrices for systematic error patterns and implement targeted improvements.

## Technical implementation best practices

**Data Pipeline Excellence:**
Implement robust audio loading with consistent sampling rates (16kHz standard), amplitude normalization across all samples, and voice activity detection for silence removal. Create data augmentation pipelines with time shifting (±10%), speed perturbation (0.9x-1.1x), and controlled noise addition (SNR 20-30dB).

**Model Architecture Guidelines:**
Start with convolutional layers for local feature detection, progress to recurrent layers for temporal modeling, and implement attention mechanisms for global dependency capture. Use residual connections for deep networks and batch normalization for stable training. Implement dropout judiciously—too little causes overfitting, too much prevents learning.

**Evaluation and Validation Framework:**
Use stratified sampling to maintain class balance across train/validation/test splits. Implement 5-fold cross-validation for hyperparameter selection. Monitor both overall accuracy and per-class performance through confusion matrices. Track training/validation loss curves for overfitting detection.

## Course context and educational objectives

Professor Lee's pedagogical approach emphasizes **progressive complexity mastery** through carefully structured baselines. The assignment develops crucial skills: understanding sequential data modeling, implementing state-of-the-art speech processing techniques, systematic debugging and optimization methods, and practical machine learning competition experience.

The homework connects to broader course themes by introducing speech processing as a gateway to more advanced topics like transformers, attention mechanisms, and self-supervised learning covered in later assignments. Students gain hands-on experience with the challenges of real-world data, the importance of systematic experimentation, and the iterative nature of machine learning development.

## Conclusion and success pathway

Success on ML2023 HW2 requires **disciplined application of systematic methodology** rather than random experimentation. Begin with solid fundamentals, progress through increasingly sophisticated techniques, and maintain rigorous experimental documentation. The assignment rewards students who understand that machine learning success comes from careful problem analysis, systematic improvement, and persistent debugging rather than hoping for algorithmic silver bullets.

Students achieving the Boss baseline demonstrate mastery of: advanced neural architectures for sequential data, systematic optimization and debugging skills, effective feature engineering for speech processing, and disciplined experimental methodology. These skills translate directly to real-world machine learning applications, making HW2 an invaluable stepping stone toward advanced AI competency.