package predictor.lstm.util.improved;

import predictor.lstm.common.HyperParameters;

public class AdaptiveLearningRateImproved {

    /**
     * Adjusts the learning rate based on the given percentage * the total
     * iterations.
     *
     * @param hyperParameters An instance of class HyperParameter
     * @return The adapted learning rate calculated using a cosine annealing
     *         strategy.
     */
    public double scheduler(HyperParameters hyperParameters) {
        var maximum = hyperParameters.getLearningRateUpperLimit();
        var minimum = hyperParameters.getLearningRateLowerLimit();
        var tCurByTmax = (double) hyperParameters.getEpochTrack() / hyperParameters.getEpoch();
        var cosineValue = Math.cos(tCurByTmax * Math.PI);
        return (minimum + 0.5 * (maximum - minimum) * (1 + cosineValue));
    }

    /**
     * Performs the Adagrad optimization step.
     * 
     * Formula: theta_{t+1} = theta_t - (eta / sqrt(G_t + epsilon)) * g_t
     * Where G_t is sum of squares of past gradients.
     *
     * @param currentWeight The current weight value.
     * @param gradient The current gradient for this weight.
     * @param accumulatedGradientSq The accumulated sum of squared gradients for this weight (G_t).
     * @param learningRate The global learning rate (eta).
     * @param epsilon A small smoothing term to avoid division by zero.
     * @return The updated weight.
     */
    public double adagradUpdate(double currentWeight, double gradient, double accumulatedGradientSq, double learningRate, double epsilon) {
        return currentWeight - (learningRate / Math.sqrt(accumulatedGradientSq + epsilon)) * gradient;
    }
}
