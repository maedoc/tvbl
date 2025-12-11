import autograd.numpy as anp
from cde import MDNEstimator, MAFEstimator, NSFEstimator, generate_test_data
import sys

def compute_quantiles(samples, q=[0.1, 0.5, 0.9]):
    # samples: (N, D)
    return anp.percentile(samples, [x * 100 for x in q], axis=0)

def main():
    print("Generating training data (Banana dataset)...")
    # Generate sufficient data for training
    params, features = generate_test_data('banana', n_samples=3000, seed=42)
    
    # Define a test condition: feature = 1.0 (middle of the 0.5-2.0 range)
    test_feature = anp.array([[1.0]])
    
    # Initialize estimators
    # Using slightly smaller networks/flows for speed in this test script, 
    # but sufficient for this simple 2D problem.
    estimators = [
        ('MDN', MDNEstimator(2, 1, n_components=5, hidden_sizes=(32, 32))),
        ('MAF', MAFEstimator(2, 1, n_flows=3, hidden_units=32)),
        ('NSF', NSFEstimator(2, 1, n_flow_steps=3, n_bins=8, hidden_features=32))
    ]
    
    results = {}
    
    for name, est in estimators:
        print(f"\nTraining {name}...")
        # Train for enough iterations to likely converge on this simple problem
        est.train(params, features, n_iter=600, learning_rate=0.005, use_tqdm=True)
        print(f"Final Loss ({name}): {est.loss_history[-1]:.4f}")
        
        # Sample
        rng = anp.random.RandomState(101)
        # Generate enough samples for stable quantiles
        samples = est.sample(test_feature, n_samples=2000, rng=rng) # (1, 2000, 2)
        samples = samples[0] # (2000, 2)
        
        # Compute Quantiles
        qs = compute_quantiles(samples) # (3, 2)
        results[name] = qs

    # --- Ground Truth Quantiles ---
    # Banana dataset logic from generate_test_data:
    # x0 ~ N(0,1)
    # x1 = z + feature * (x0^2 - 2)
    # where z ~ N(0,1)
    # for feature = 1.0
    rng_gt = anp.random.RandomState(42)
    x0 = rng_gt.randn(10000)
    z = rng_gt.randn(10000)
    x1 = z + 1.0 * (x0**2 - 2.0)
    gt_samples = anp.vstack([x0, x1]).T
    results['GroundTruth'] = compute_quantiles(gt_samples)
    
    # --- Report ---
    print("\n" + "="*80)
    print("QUANTILE COMPARISON (Condition: feature=1.0)")
    print("Target Distribution: Banana Shape")
    print("Quantiles: 10%, 50% (Median), 90%")
    print("="*80)
    
    header = f"{ 'Model':<12} | { 'Param 0 (x)':<30} | { 'Param 1 (y)':<30}"
    print(header)
    print("-" * len(header))
    
    for name in ['GroundTruth', 'MDN', 'MAF', 'NSF']:
        q = results[name]
        # Format: 10%, 50%, 90%
        p0_str = f"{q[0,0]:5.2f}, {q[1,0]:5.2f}, {q[2,0]:5.2f}"
        p1_str = f"{q[0,1]:5.2f}, {q[1,1]:5.2f}, {q[2,1]:5.2f}"
        print(f"{name:<12} | {p0_str:<30} | {p1_str:<30}")
    print("="*80)

if __name__ == "__main__":
    main()
