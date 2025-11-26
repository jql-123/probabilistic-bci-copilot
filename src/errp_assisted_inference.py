"""
PHASE 2 - TASK 1: ErrP-Assisted Inference Pipeline

Implements the complete ErrP workflow:
1. Initial decision (500ms motion only)
2. ErrP simulation (250ms delay)
3. Belief update with ErrP observation
4. Second decision (if needed)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.append('..')
sys.path.append('../../')
from decorrelated_task_loader import DecorrelatedTaskLoader
from train_test_split_evaluation import ProperTrainTestCopilot

@dataclass
class ErrPEvent:
    """ErrP event with timing and detection info."""
    timestep: int
    clicked_target: int
    true_target: int
    is_error: bool
    detected: bool
    confidence: float

@dataclass
class ErrPDecisionStep:
    """Single decision step in ErrP workflow."""
    step_name: str
    timestep: int
    belief_distribution: np.ndarray
    max_belief: float
    predicted_target: Optional[int]
    confidence: float
    uncertainty: float
    should_click: bool

class ErrPAssistedCopilot:
    """Spatial copilot with ErrP-assisted inference workflow."""

    def __init__(self, clean_copilot, subject_id: str = "H1"):
        self.copilot = clean_copilot
        self.subject_id = subject_id

        # ErrP parameters (from literature)
        self.errp_detection_rate = 0.70  # 70% for errors
        self.errp_false_positive_rate = 0.08  # 8% false positives
        self.errp_delay_samples = 12  # 250ms at 50Hz

        # Decision thresholds
        self.initial_click_threshold = 0.5  # Click if belief > 50%
        self.retry_threshold = 0.6  # Retry if new belief > 60%

    def simulate_errp_signal(self, clicked_target: int, true_target: int,
                           timestep: int) -> ErrPEvent:
        """Simulate realistic ErrP response."""
        is_error = (clicked_target != true_target)

        if is_error:
            # Error occurred: 70% detection rate
            detected = np.random.random() < self.errp_detection_rate
            confidence = 0.6 + 0.3 * np.random.random()  # 60-90%
        else:
            # Correct: 8% false positive rate
            detected = np.random.random() < self.errp_false_positive_rate
            confidence = 0.1 + 0.2 * np.random.random()  # 10-30%

        return ErrPEvent(
            timestep=timestep,
            clicked_target=clicked_target,
            true_target=true_target,
            is_error=is_error,
            detected=detected,
            confidence=confidence
        )

    def run_errp_workflow(self, neural_features: List[np.ndarray],
                         cursor_positions: List[np.ndarray],
                         true_target: int, verbose: bool = True) -> Dict:
        """
        Complete ErrP-assisted inference workflow.

        Returns detailed step-by-step results showing belief evolution.
        """

        if verbose:
            print(f"\n=== ErrP WORKFLOW: True target = {true_target} ===")

        workflow_steps = []
        total_samples = len(neural_features)

        # === STEP 1: Initial Motion Period (0-500ms = 25 samples) ===
        initial_window = min(25, total_samples)
        initial_neural = neural_features[:initial_window]
        initial_cursor = cursor_positions[:initial_window]

        # Run inference without ErrP
        initial_decision = self.copilot.make_decision(initial_neural, initial_cursor)

        step1 = ErrPDecisionStep(
            step_name="Initial (Motion Only)",
            timestep=initial_window,
            belief_distribution=initial_decision.target_probabilities.copy(),
            max_belief=np.max(initial_decision.target_probabilities),
            predicted_target=initial_decision.target_id if initial_decision.should_assist else None,
            confidence=initial_decision.confidence,
            uncertainty=initial_decision.uncertainty,
            should_click=initial_decision.should_assist and initial_decision.confidence > self.initial_click_threshold
        )
        workflow_steps.append(step1)

        if verbose:
            print(f"\\nStep 1 - Initial Decision (t={initial_window}):")
            print(f"  Max belief: {step1.max_belief:.3f}")
            print(f"  Predicted: {step1.predicted_target}")
            print(f"  Should click: {step1.should_click}")
            print(f"  Belief: {[f'{b:.3f}' for b in step1.belief_distribution[[0,2,4,6]]]}")

        # === STEP 2: Click Decision ===
        clicked_target = None
        errp_event = None

        if step1.should_click and step1.predicted_target is not None:
            clicked_target = step1.predicted_target

            # === STEP 3: ErrP Response (250ms delay) ===
            errp_timestep = initial_window + self.errp_delay_samples
            errp_event = self.simulate_errp_signal(clicked_target, true_target, errp_timestep)

            if verbose:
                print(f"\\nStep 2 - Click Decision:")
                print(f"  Clicked target: {clicked_target}")
                print(f"  True target: {true_target}")
                print(f"  Is error: {errp_event.is_error}")
                print(f"\\nStep 3 - ErrP Response (t={errp_timestep}):")
                print(f"  ErrP detected: {errp_event.detected}")
                print(f"  ErrP confidence: {errp_event.confidence:.3f}")

            # === STEP 4: Extended Window with ErrP (500ms + 250ms = 750ms = 37 samples) ===
            extended_window = min(37, total_samples)
            extended_neural = neural_features[:extended_window]
            extended_cursor = cursor_positions[:extended_window]

            # Run inference WITH ErrP observation
            errp_decision = self.copilot.make_decision(
                extended_neural, extended_cursor,
                clicked_target, errp_event.detected
            )

            step4 = ErrPDecisionStep(
                step_name="After ErrP Integration",
                timestep=extended_window,
                belief_distribution=errp_decision.target_probabilities.copy(),
                max_belief=np.max(errp_decision.target_probabilities),
                predicted_target=errp_decision.target_id if errp_decision.should_assist else None,
                confidence=errp_decision.confidence,
                uncertainty=errp_decision.uncertainty,
                should_click=errp_decision.should_assist
            )
            workflow_steps.append(step4)

            if verbose:
                print(f"\\nStep 4 - After ErrP Integration (t={extended_window}):")
                print(f"  New max belief: {step4.max_belief:.3f}")
                print(f"  New predicted: {step4.predicted_target}")
                print(f"  Uncertainty change: {step1.uncertainty:.3f} → {step4.uncertainty:.3f}")
                print(f"  New belief: {[f'{b:.3f}' for b in step4.belief_distribution[[0,2,4,6]]]}")

            # === STEP 5: Retry Decision ===
            should_retry = False
            retry_target = None

            if (step4.predicted_target != clicked_target and
                step4.predicted_target is not None and
                step4.confidence > self.retry_threshold):

                should_retry = True
                retry_target = step4.predicted_target

                if verbose:
                    print(f"\\nStep 5 - Retry Decision:")
                    print(f"  Should retry: {should_retry}")
                    print(f"  Retry target: {retry_target}")
                    print(f"  Reason: New belief {step4.confidence:.3f} > threshold {self.retry_threshold}")

            # Determine final decision
            final_target = retry_target if should_retry else clicked_target
            final_correct = (final_target == true_target)

        else:
            # No initial click made
            final_target = None
            final_correct = False
            should_retry = False
            retry_target = None

            if verbose:
                print(f"\\nNo initial click made (belief too low)")

        # === RESULTS SUMMARY ===
        if verbose:
            print(f"\\n=== WORKFLOW SUMMARY ===")
            print(f"Initial prediction: {step1.predicted_target}")
            print(f"Clicked target: {clicked_target}")
            print(f"ErrP detected: {errp_event.detected if errp_event else 'N/A'}")
            print(f"Should retry: {should_retry}")
            print(f"Final target: {final_target}")
            print(f"Correct: {'✅' if final_correct else '❌'}")

        return {
            'workflow_steps': workflow_steps,
            'initial_prediction': step1.predicted_target,
            'clicked_target': clicked_target,
            'errp_event': errp_event,
            'should_retry': should_retry,
            'retry_target': retry_target,
            'final_target': final_target,
            'final_correct': final_correct,
            'true_target': true_target
        }

    def visualize_belief_evolution(self, workflow_result: Dict, save_path: str = None):
        """Visualize belief evolution through ErrP workflow."""

        workflow_steps = workflow_result['workflow_steps']
        true_target = workflow_result['true_target']

        fig, axes = plt.subplots(2, len(workflow_steps), figsize=(5*len(workflow_steps), 8))
        if len(workflow_steps) == 1:
            axes = axes.reshape(-1, 1)

        for i, step in enumerate(workflow_steps):
            # Top subplot: All 8 targets
            ax_top = axes[0, i]
            beliefs = step.belief_distribution

            # Color code: blue for used targets, red for unused
            colors = ['blue' if j in [0,2,4,6] else 'red' for j in range(8)]

            bars = ax_top.bar(range(8), beliefs, color=colors)

            # Set individual alpha values
            for bar, j in zip(bars, range(8)):
                alpha = 0.8 if j in [0,2,4,6] else 0.4
                bar.set_alpha(alpha)

            # Highlight true target
            if true_target is not None:
                bars[true_target].set_edgecolor('gold')
                bars[true_target].set_linewidth(3)

            # Highlight predicted target
            if step.predicted_target is not None:
                bars[step.predicted_target].set_hatch('///')

            ax_top.set_ylim(0, 1)
            ax_top.set_xlabel('Target ID')
            ax_top.set_ylabel('Belief Probability')
            ax_top.set_title(f'{step.step_name}\\nt={step.timestep}')
            ax_top.grid(True, alpha=0.3)

            # Add belief values on bars
            for j, (bar, belief) in enumerate(zip(bars, beliefs)):
                if belief > 0.01:  # Only show significant beliefs
                    ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{belief:.2f}', ha='center', va='bottom', fontsize=8)

            # Bottom subplot: Only used targets {0,2,4,6}
            ax_bottom = axes[1, i]
            used_indices = [0, 2, 4, 6]
            used_beliefs = beliefs[used_indices]
            used_colors = ['blue'] * 4

            bars_used = ax_bottom.bar(range(4), used_beliefs, color=used_colors, alpha=0.8)
            ax_bottom.set_xticks(range(4))
            ax_bottom.set_xticklabels([f'T{j}' for j in used_indices])
            ax_bottom.set_ylim(0, 1)
            ax_bottom.set_xlabel('Used Targets Only')
            ax_bottom.set_ylabel('Belief Probability')
            ax_bottom.set_title(f'Max: {step.max_belief:.3f}\\nPred: T{step.predicted_target}')
            ax_bottom.grid(True, alpha=0.3)

            # Highlight true target in bottom plot
            if true_target in used_indices:
                true_idx_in_used = used_indices.index(true_target)
                bars_used[true_idx_in_used].set_edgecolor('gold')
                bars_used[true_idx_in_used].set_linewidth(3)

            # Highlight predicted target in bottom plot
            if step.predicted_target in used_indices:
                pred_idx_in_used = used_indices.index(step.predicted_target)
                bars_used[pred_idx_in_used].set_hatch('///')

        plt.suptitle(f'ErrP Workflow: Belief Evolution (True Target: T{true_target})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Belief evolution plot saved: {save_path}")

        return fig

def test_errp_workflow():
    """Test the ErrP workflow on a single trial."""

    print("="*60)
    print("TASK 1: ErrP WORKFLOW IMPLEMENTATION")
    print("="*60)

    # Setup
    evaluator = ProperTrainTestCopilot("H1")
    train_trials, test_trials = evaluator.train_test_split(test_size=0.3)
    training_means = evaluator.compute_training_means(train_trials)
    clean_copilot = evaluator.create_uncontaminated_copilot(training_means)

    # Create ErrP-assisted copilot
    errp_copilot = ErrPAssistedCopilot(clean_copilot, "H1")

    # Test on a single trial
    test_trial = test_trials[0]
    decorr_target = test_trial['target_id']
    spatial_target = evaluator.target_mapping[decorr_target]

    print(f"\\nTesting on Trial 1:")
    print(f"  Decorrelated target: {decorr_target}")
    print(f"  Spatial target: {spatial_target}")
    print(f"  Trial length: {len(test_trial['neural_features'])} timesteps")

    # Extract trial data
    neural_features = [feat.astype(np.float64) for feat in test_trial['neural_features'][:50]]
    cursor_positions = [pos.astype(np.float64) for pos in test_trial['positions'][:50]]

    # Run ErrP workflow
    np.random.seed(42)  # Reproducible ErrP simulation
    workflow_result = errp_copilot.run_errp_workflow(
        neural_features, cursor_positions, spatial_target, verbose=True
    )

    # Visualize results
    fig = errp_copilot.visualize_belief_evolution(
        workflow_result, 'errp_workflow_belief_evolution.png'
    )

    print(f"\\n" + "="*60)
    print("✅ TASK 1 COMPLETE: ErrP Workflow Implemented")
    print("="*60)
    print("✅ Multi-step belief evolution")
    print("✅ Realistic ErrP simulation")
    print("✅ Click and retry logic")
    print("✅ Detailed step tracking")

    return workflow_result, errp_copilot

if __name__ == "__main__":
    workflow_result, errp_copilot = test_errp_workflow()