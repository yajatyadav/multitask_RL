import jax
import jax.numpy as jnp
import distrax

def diagnose_base_distribution():
    """Diagnose the base distribution issue."""
    
    action_dim = 3
    batch_size = 5
    
    # Your setup
    means = jnp.zeros((batch_size, action_dim))
    log_stds = jnp.ones((batch_size, action_dim)) * (-1.0)
    temperature = 1.0
    
    print("="*60)
    print("DIAGNOSING BASE DISTRIBUTION")
    print("="*60)
    print(f"Means shape: {means.shape}")
    print(f"Log stds shape: {log_stds.shape}")
    print(f"Scale diag shape: {jnp.exp(log_stds).shape}")
    
    base_dist = distrax.MultivariateNormalDiag(
        loc=means, 
        scale_diag=jnp.exp(log_stds) * temperature
    )
    
    print(f"\nBase distribution batch shape: {base_dist.batch_shape}")
    print(f"Base distribution event shape: {base_dist.event_shape}")
    
    # Test point
    x = jnp.array([-0.18383715, -0.06265909, -0.12324166])
    
    print(f"\nTest point x: {x}")
    print(f"Test point shape: {x.shape}")
    
    # Compute log prob for this single point
    # Method 1: Direct call (what shape does it expect?)
    try:
        lp1 = base_dist.log_prob(x)
        print(f"\nlog_prob(x) - shape {x.shape}:")
        print(f"  Result shape: {lp1.shape}")
        print(f"  Result: {lp1}")
    except Exception as e:
        print(f"\nlog_prob(x) failed: {e}")
    
    # Method 2: With batch dimension
    try:
        lp2 = base_dist.log_prob(x[None, :])
        print(f"\nlog_prob(x[None, :]) - shape {x[None, :].shape}:")
        print(f"  Result shape: {lp2.shape}")
        print(f"  Result: {lp2}")
    except Exception as e:
        print(f"\nlog_prob(x[None, :]) failed: {e}")
    
    # Method 3: Replicated across batch
    x_batched = jnp.tile(x[None, :], (batch_size, 1))
    try:
        lp3 = base_dist.log_prob(x_batched)
        print(f"\nlog_prob(x_batched) - shape {x_batched.shape}:")
        print(f"  Result shape: {lp3.shape}")
        print(f"  Result: {lp3}")
    except Exception as e:
        print(f"\nlog_prob(x_batched) failed: {e}")
    
    # Manual calculation for a single Gaussian
    print("\n" + "="*60)
    print("MANUAL CALCULATION (single Gaussian)")
    print("="*60)
    mean_single = means[0]
    std_single = jnp.exp(log_stds[0]) * temperature
    
    # Log prob formula: -0.5 * sum((x - mu)^2 / sigma^2) - sum(log(sigma)) - 0.5 * k * log(2*pi)
    k = action_dim
    diff = x - mean_single
    log_prob_manual = (
        -0.5 * jnp.sum((diff ** 2) / (std_single ** 2))
        - jnp.sum(jnp.log(std_single))
        - 0.5 * k * jnp.log(2 * jnp.pi)
    )
    
    print(f"Mean: {mean_single}")
    print(f"Std: {std_single}")
    print(f"x: {x}")
    print(f"Manual log prob: {log_prob_manual:.6f}")
    
    # Compare with single distribution
    single_dist = distrax.MultivariateNormalDiag(
        loc=mean_single,
        scale_diag=std_single
    )
    single_lp = single_dist.log_prob(x)
    print(f"Single dist log prob: {single_lp:.6f}")
    print(f"Match? {jnp.allclose(log_prob_manual, single_lp)}")


if __name__ == "__main__":
    diagnose_base_distribution()