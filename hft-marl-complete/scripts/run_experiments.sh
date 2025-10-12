#!/bin/bash

# Multi-Agent High-Frequency Trading Experiment Runner
# ====================================================

set -e  # Exit on any error

echo "🚀 Multi-Agent High-Frequency Trading Experiment Runner"
echo "======================================================="

# Configuration
ALGORITHMS=("maddpg" "mappo")
EPISODES=10000
EVAL_EPISODES=100
SEEDS=(42 123 456)

# Create directories
mkdir -p data/features models results logs mlruns

# Function to run training
run_training() {
    local algorithm=$1
    local episodes=$2
    local seed=$3
    
    echo "📊 Training $algorithm with $episodes episodes (seed: $seed)"
    echo "--------------------------------------------------------"
    
    python main.py train \
        --algorithm $algorithm \
        --episodes $episodes \
        --seed $seed \
        --device cuda
    
    echo "✅ Training completed for $algorithm (seed: $seed)"
    echo ""
}

# Function to run evaluation
run_evaluation() {
    local algorithm=$1
    local seed=$2
    
    echo "📈 Evaluating $algorithm (seed: $seed)"
    echo "--------------------------------------"
    
    python main.py evaluate \
        --model models/${algorithm}_final.pt \
        --episodes $EVAL_EPISODES
    
    echo "✅ Evaluation completed for $algorithm (seed: $seed)"
    echo ""
}

# Function to run baseline
run_baseline() {
    local strategy=$1
    
    echo "📉 Running baseline: $strategy"
    echo "-------------------------------"
    
    python main.py baseline \
        --strategy $strategy \
        --episodes $EVAL_EPISODES
    
    echo "✅ Baseline completed: $strategy"
    echo ""
}

# Function to run comparison
run_comparison() {
    echo "⚖️  Comparing all models"
    echo "========================"
    
    python main.py compare \
        --models models/maddpg_final.pt models/mappo_final.pt \
        --episodes $EVAL_EPISODES
    
    echo "✅ Comparison completed"
    echo ""
}

# Main execution
main() {
    echo "Starting comprehensive experiment suite..."
    echo ""
    
    # Create dummy data if not exists
    if [ ! -f "data/features/dev_tensors.npz" ]; then
        echo "📁 Creating dummy data..."
        python main.py train --episodes 1 --seed 42
        echo "✅ Dummy data created"
        echo ""
    fi
    
    # Run training for all algorithms and seeds
    for algorithm in "${ALGORITHMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_training $algorithm $EPISODES $seed
        done
    done
    
    # Run evaluation for all algorithms
    for algorithm in "${ALGORITHMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_evaluation $algorithm $seed
        done
    done
    
    # Run baseline strategies
    BASELINES=("avellaneda-stoikov" "twap" "vwap" "pov" "momentum" "mean-reversion" "technical-analysis" "random-walk" "buy-and-hold")
    
    for baseline in "${BASELINES[@]}"; do
        run_baseline $baseline
    done
    
    # Run model comparison
    run_comparison
    
    # Generate final report
    echo "📋 Generating final report..."
    echo "============================"
    
    # Create summary report
    cat > results/experiment_summary.md << EOF
# Multi-Agent High-Frequency Trading Experiment Summary

## Experiment Configuration
- Algorithms: ${ALGORITHMS[*]}
- Training Episodes: $EPISODES
- Evaluation Episodes: $EVAL_EPISODES
- Seeds: ${SEEDS[*]}
- Date: $(date)

## Results
Results are stored in the following directories:
- Models: \`models/\`
- Results: \`results/\`
- Logs: \`logs/\`
- MLflow: \`mlruns/\`

## Next Steps
1. Review the results in the \`results/\` directory
2. Check MLflow dashboard for experiment tracking
3. Analyze performance metrics and comparisons
4. Select best performing model for deployment

## Files Generated
- \`models/maddpg_final.pt\`: Trained MADDPG model
- \`models/mappo_final.pt\`: Trained MAPPO model
- \`results/*_results.json\`: Detailed results for each experiment
- \`results/*_report.txt\`: Comprehensive evaluation reports
- \`logs/*.log\`: Training logs
EOF
    
    echo "✅ Final report generated: results/experiment_summary.md"
    echo ""
    
    # Display summary
    echo "🎉 All experiments completed successfully!"
    echo "=========================================="
    echo "📊 Results available in: results/"
    echo "🤖 Models available in: models/"
    echo "📝 Logs available in: logs/"
    echo "📈 MLflow dashboard: mlruns/"
    echo ""
    echo "To view results:"
    echo "  - Check results/experiment_summary.md for overview"
    echo "  - Run 'mlflow ui' to view experiment dashboard"
    echo "  - Review individual result files in results/ directory"
}

# Handle command line arguments
case "${1:-all}" in
    "train")
        echo "Running training only..."
        for algorithm in "${ALGORITHMS[@]}"; do
            run_training $algorithm $EPISODES 42
        done
        ;;
    "eval")
        echo "Running evaluation only..."
        for algorithm in "${ALGORITHMS[@]}"; do
            run_evaluation $algorithm 42
        done
        ;;
    "baseline")
        echo "Running baselines only..."
        BASELINES=("avellaneda-stoikov" "twap" "momentum" "mean-reversion")
        for baseline in "${BASELINES[@]}"; do
            run_baseline $baseline
        done
        ;;
    "compare")
        echo "Running comparison only..."
        run_comparison
        ;;
    "all"|*)
        main
        ;;
esac

echo "🏁 Experiment runner finished!"
