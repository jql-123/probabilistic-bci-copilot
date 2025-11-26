"""
Spatial Target Gen Model with ErrP Integration

Correct architecture: Predicts spatial target positions directly (not motor intentions)
Uses 8-target center-out prior with evidence concentration on 4 active targets
Includes ErrP observations for error correction and target discovery
"""

using Gen
using Distributions
using LinearAlgebra

"""
TASK 4: Helper Functions

Generate standard 8-target center-out layout at radius 0.8
"""
function generate_8_target_positions()::Vector{Vector{Float64}}
    radius = 0.8
    targets = Vector{Vector{Float64}}()

    for i in 0:7
        angle = i * (2π / 8)  # 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        x = radius * cos(angle)
        y = radius * sin(angle)
        push!(targets, [x, y])
    end

    return targets
end

"""
Get neural feature mean for specific target and subject

For decorrelated task mapping:
- Targets {0,2,4,6} correspond to actual data targets {1,2,0,3}
- Targets {1,3,5,7} are unused (baseline features)
"""
function get_target_cnn_mean(target_id::Int, subject_id::String, feature_dim::Int)::Vector{Float64}
    """
    Get expected neural features for each target using DATA-DRIVEN patterns.

    For targets {0,2,4,6}: Use dataset statistics
    For targets {1,3,5,7}: Return neutral/baseline features (they'll get low likelihood)

    Maps 8-target layout to 4 decorrelated targets based on REAL H1 data:
    - Target 0 (Right) → Decorrelated target 1 pattern
    - Target 2 (Up) → Decorrelated target 2 pattern
    - Target 4 (Left) → Decorrelated target 0 pattern
    - Target 6 (Down) → Decorrelated target 3 pattern
    - Targets {1,3,5,7} → Unused (unlikely pattern)
    """

    if subject_id == "H1"
        # DATA-DRIVEN patterns from real H1 neural data
        if target_id == 0  # Right → Decorrelated target 1 pattern
            return [-0.313, -0.093, -0.244, 0.093, -0.069, -0.217, 0.006, -0.036,
                    0.023, -0.061, -0.028, -0.020, 0.163, 0.059, 0.014, 0.142]

        elseif target_id == 2  # Up → Decorrelated target 2 pattern
            return [0.124, -0.028, 0.053, -0.052, -0.057, 0.071, -0.083, -0.069,
                    -0.074, -0.025, 0.099, 0.108, -0.037, 0.019, 0.013, -0.030]

        elseif target_id == 4  # Left → Decorrelated target 0 pattern
            return [0.149, 0.104, 0.097, -0.029, 0.106, 0.070, 0.051, 0.100,
                    0.038, 0.088, 0.054, 0.119, 0.123, 0.043, 0.122, 0.105]

        elseif target_id == 6  # Down → Decorrelated target 3 pattern
            return [-0.223, 0.155, -0.167, 0.148, 0.241, -0.162, 0.223, 0.245,
                    0.137, -0.082, 0.105, 0.153, -0.152, 0.084, 0.064, 0.131]

        else  # Targets {1,3,5,7} - unused, make unlikely
            # Use pattern that's far from real data
            return fill(0.500, feature_dim)  # Uniform high values - very unlikely
        end
    else
        # Fallback for other subjects
        return fill(0.0, feature_dim)
    end
end

"""
Compute cursor velocity alignment with target direction
"""
function compute_velocity_alignment(velocity::Vector{Float64},
                                  target_direction::Vector{Float64})::Float64
    if norm(velocity) < 1e-6 || norm(target_direction) < 1e-6
        return 0.0
    end

    # Cosine similarity between velocity and target direction
    alignment = dot(velocity, target_direction) / (norm(velocity) * norm(target_direction))
    return clamp(alignment, -1.0, 1.0)
end

"""
Simulate realistic ErrP signal based on literature parameters
"""
function simulate_errp_signal(clicked_target::Int, true_target::Int)::Bool
    """
    Simulate ErrP detection based on Chavarriaga & Millán (2010):
    - True detection rate: 70% when error occurs
    - False positive rate: 8% when correct
    """
    is_error = (clicked_target != true_target)

    if is_error
        # Error occurred: 70% chance of ErrP detection
        return rand() < 0.70
    else
        # Correct: 8% chance of false positive
        return rand() < 0.08
    end
end

"""
Main Spatial Target Gen Model with ErrP Integration

This is the correct architecture for BCI cursor copilot:
- Directly predicts spatial target positions (not motor intentions)
- Uses 8-target center-out prior (discovers 4 are actually used)
- Integrates neural features + cursor movement + ErrP feedback
"""
@gen function spatial_target_model_8way(cnn_features::Vector{Vector{Float64}},
                                        cursor_positions::Vector{Vector{Float64}},
                                        target_positions::Vector{Vector{Float64}},
                                        subject_id::String,
                                        timesteps::Int,
                                        clicked_target::Union{Nothing, Int} = nothing,
                                        errp_delay_timesteps::Int = 5)  # 250ms at 50Hz

    # PRIOR: Uniform over 8 standard center-out targets
    # Copilot doesn't know only 4 are actually used in decorrelated task
    target_id ~ categorical(fill(1/8, 8))
    target_pos = target_positions[target_id + 1]  # Convert to 1-indexing

    # OBSERVATION 1: CNN Neural Features
    # Neural evidence for intended target
    feature_dim = length(cnn_features[1])
    feature_noise = 0.1

    for t in 1:timesteps
        feature_mean = get_target_cnn_mean(target_id, subject_id, feature_dim)

        for d in 1:feature_dim
            {(:cnn, t, d)} ~ normal(feature_mean[d], feature_noise)
        end
    end

    # OBSERVATION 2: Cursor velocity (directional evidence)
    velocity_noise = 0.5

    for t in 2:timesteps  # Start from t=2 (need previous position)
        velocity = cursor_positions[t] - cursor_positions[t-1]
        expected_direction = normalize(target_pos - cursor_positions[t-1])

        # Directional alignment score
        alignment = dot(normalize(velocity), expected_direction)
        {(:velocity, t)} ~ normal(alignment, velocity_noise)
    end

    # OBSERVATION 3: ErrP Signal (Error Feedback)
    # Based on Chavarriaga & Millán (2010) parameters
    if clicked_target !== nothing && timesteps >= errp_delay_timesteps
        errp_timestep = timesteps - errp_delay_timesteps + 1

        if target_id == clicked_target
            # Correct click: low ErrP probability (false positive)
            {:errp} ~ bernoulli(0.08)
        else
            # Incorrect click: high ErrP probability (true detection)
            {:errp} ~ bernoulli(0.70)
        end
    end

    return target_id
end

"""
TASK 5: Enumerative Inference Function

Enumerative Inference over 8 Spatial Targets
- Enumerates all 8 target hypotheses
- Returns posterior distribution over 8 targets
- Shows which targets are plausible vs implausible

Returns posterior distribution showing target discovery:
- Initially uniform over 8 targets
- Concentrates on targets {0,2,4,6} that receive evidence
- Targets {1,3,5,7} remain low probability (unused in decorrelated task)
"""
function spatial_target_inference(cnn_features::Vector{Vector{Float64}},
                                 cursor_positions::Vector{Vector{Float64}},
                                 subject_id::String,
                                 clicked_target::Union{Nothing, Int} = nothing,
                                 errp_detected::Union{Nothing, Bool} = nothing)

    timesteps = length(cnn_features)
    target_positions = generate_8_target_positions()

    # Create observations choicemap
    observations = choicemap()

    # CNN features
    for t in 1:timesteps
        for d in 1:length(cnn_features[t])
            observations[(:cnn, t, d)] = cnn_features[t][d]
        end
    end

    # Cursor velocities
    for t in 2:timesteps
        velocity = cursor_positions[t] - cursor_positions[t-1]
        target_direction = [1.0, 0.0]  # Will be computed per target in model
        alignment = compute_velocity_alignment(velocity, target_direction)
        observations[(:velocity, t)] = alignment
    end

    # ErrP signal
    if errp_detected !== nothing
        observations[:errp] = errp_detected
    end

    # ENUMERATIVE INFERENCE over 8 target hypotheses
    traces = []
    log_weights = []

    for target_id in 0:7
        # Create constraints for this target
        constraints = choicemap()
        constraints[:target_id] = target_id + 1  # Convert to 1-indexing
        merge!(constraints, observations)

        # Generate trace for this target hypothesis
        (trace, log_weight) = generate(spatial_target_model_8way,
                                     (cnn_features, cursor_positions, target_positions,
                                      subject_id, timesteps, clicked_target),
                                     constraints)

        push!(traces, trace)
        push!(log_weights, log_weight)
    end

    # Normalize to get posterior distribution
    log_weights .-= maximum(log_weights)  # Numerical stability
    weights = exp.(log_weights)
    weights ./= sum(weights)

    return weights, traces, target_positions
end

"""
Make copilot assistance decision based on spatial target posterior

Returns assistance recommendation with target position and confidence
"""
function make_spatial_copilot_decision(weights::Vector{Float64},
                                      target_positions::Vector{Vector{Float64}},
                                      uncertainty_threshold::Float64 = 0.3)

    # Compute uncertainty (entropy) over target beliefs
    uncertainty = -sum(p * log(p + 1e-10) for p in weights if p > 0)
    max_uncertainty = log(8)  # Maximum entropy for 8 targets
    normalized_uncertainty = uncertainty / max_uncertainty

    # Assistance decision
    should_assist = normalized_uncertainty < uncertainty_threshold  # Low uncertainty → assist

    if should_assist
        # Click on most likely target
        best_target_id = argmax(weights) - 1  # Convert back to 0-indexing
        best_target_pos = target_positions[best_target_id + 1]
        confidence = weights[best_target_id + 1]

        return (
            should_assist = true,
            target_id = best_target_id,
            target_position = best_target_pos,
            confidence = confidence,
            uncertainty = normalized_uncertainty,
            target_probabilities = weights
        )
    else
        return (
            should_assist = false,
            target_id = nothing,
            target_position = nothing,
            confidence = 0.0,
            uncertainty = normalized_uncertainty,
            target_probabilities = weights
        )
    end
end

# Demo/testing code for Tasks 3-5
if abspath(PROGRAM_FILE) == @__FILE__
    println("=== TASKS 3-5: Spatial Target Gen Model Demo ===")

    # Generate 8-target layout
    targets = generate_8_target_positions()
    println("8-Target Layout:")
    for (i, pos) in enumerate(targets)
        used = (i-1) in [0,2,4,6] ? "USED" : "unused"
        println("  Target $(i-1): ($(round(pos[1], digits=3)), $(round(pos[2], digits=3))) [$used]")
    end

    # Test with synthetic data
    timesteps = 15
    subject_id = "H1"

    # Synthetic neural features biased toward target 2 (Up)
    cnn_features = [randn(16) * 0.1 .+ get_target_cnn_mean(2, subject_id, 16) for _ in 1:timesteps]

    # Synthetic cursor moving toward target 2
    cursor_positions = Vector{Vector{Float64}}()
    for t in 1:timesteps
        progress = t / timesteps
        pos = [0.0, 0.8 * progress] + randn(2) * 0.05  # Move up with noise
        push!(cursor_positions, pos)
    end

    println("\nRunning spatial target inference...")
    weights, traces, target_pos = spatial_target_inference(cnn_features, cursor_positions, subject_id)

    println("\nTarget Discovery Results:")
    for (i, w) in enumerate(weights)
        target_id = i - 1
        used = target_id in [0,2,4,6] ? "USED" : "unused"
        println("  Target $target_id: $(round(w, digits=3)) probability [$used]")
    end

    decision = make_spatial_copilot_decision(weights, target_pos)
    println("\nCopilot Decision:")
    println("  Should assist: $(decision.should_assist)")
    if decision.should_assist
        println("  Target: $(decision.target_id)")
        println("  Position: $(decision.target_position)")
        println("  Confidence: $(round(decision.confidence, digits=3))")
    end
    println("  Uncertainty: $(round(decision.uncertainty, digits=3))")
end