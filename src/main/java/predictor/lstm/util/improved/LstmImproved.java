package predictor.lstm.util.improved;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.stream.IntStream;
import predictor.lstm.util.MatrixWeight;

public class LstmImproved {

    private double[] inputData;
    private double outputData;
    
    // Gradients
    private double derivativeLWrtRi = 0;
    private double derivativeLWrtRo = 0;
    private double derivativeLWrtRz = 0;
    private double derivativeLWrtWi = 0;
    private double derivativeLWrtWo = 0;
    private double derivativeLWrtWz = 0;
    
    // Accumulated Squared Gradients for Adagrad
    // 0:Wi, 1:Wo, 2:Wz, 3:Ri, 4:Ro, 5:Rz
    private ArrayList<Double> accumulatedGradientSq; 

    private double learningRate;
    private int epoch = 100;
    private double epsilon = 1e-8;

    protected ArrayList<CellImproved> cells;

    public LstmImproved(LstmBuilder builder) {
        this.inputData = builder.inputData;
        this.outputData = builder.outputData;
        this.learningRate = builder.learningRate;
        this.epoch = builder.epoch;
        
        // Initialize Adagrad state (6 parameters per cell-layer level)
        // Since parameters are shared across time steps (BPTT), we have one accumulator per parameter type.
        this.accumulatedGradientSq = new ArrayList<>();
        for(int i=0; i<6; i++) {
            this.accumulatedGradientSq.add(0.0);
        }
    }

    /**
     * Forward propagation with dropout regularization.
     */
    public void forwardprop(boolean training) {
        try {
            double dropoutRate = 0.2; 

            for (int i = 0; i < this.cells.size(); i++) {
                if (training) {
                    this.cells.get(i).setDropoutEnabled(true);
                    this.cells.get(i).setDropoutRate(dropoutRate);
                } else {
                    this.cells.get(i).setDropoutEnabled(false);
                }

                this.cells.get(i).forwardPropogation();

                if (i < this.cells.size() - 1) {
                    this.cells.get(i + 1).setYtMinusOne(this.cells.get(i).getYt());
                    this.cells.get(i + 1).setCtMinusOne(this.cells.get(i).getCt());
                    
                    // Set error for intermediate steps based on next step's input?
                    // Original code: this.cells.get(i).setError(this.cells.get(i).getYt() - this.cells.get(i + 1).getXt());
                    // This is for Sequence-to-Sequence where next input is current output?
                    // In many-to-one or standard training, we usually only care about final error.
                    // However, if this is strictly predicting next time step at EACH step:
                    // input: x1, x2, x3
                    // target: x2, x3, x4
                    // Then error at step 1 is y1 - x2.
                    // The original code uses `cells.get(i+1).getXt()` as target for `cell[i]`.
                    // This implies the task is "predict the next input in the sequence".
                    // I will preserve this logic as it defines the training objective.
                     this.cells.get(i).setError(
                            this.cells.get(i).getYt() - this.cells.get(i + 1).getXt());
                } else {
                     // Last cell error is against final outputData
                     this.cells.get(i).setError(this.cells.get(i).getYt() - this.outputData);
                }
            }
        } catch (IndexOutOfBoundsException e) {
            e.printStackTrace();
        }
    }

    /**
     * Backward propagation.
     */
    public void backwardprop() {
        // Reset gradients accumulation for this pass
        this.derivativeLWrtRi = 0;
        this.derivativeLWrtRo = 0;
        this.derivativeLWrtRz = 0;
        this.derivativeLWrtWi = 0;
        this.derivativeLWrtWo = 0;
        this.derivativeLWrtWz = 0;

        for (int i = this.cells.size() - 1; i >= 0; i--) {
            if (i < this.cells.size() - 1) {
                // Pass gradient of Cell State from future to past
                this.cells.get(i).setDlByDc(this.cells.get(i + 1).getDlByDc());
            }
            this.cells.get(i).backwardPropogation();
        }

        // Accumulate gradients for weights (shared weights)
        for (int i = 0; i < this.cells.size(); i++) {
            this.derivativeLWrtRi += this.cells.get(i).getYtMinusOne() * this.cells.get(i).getDelI();
            this.derivativeLWrtRo += this.cells.get(i).getYtMinusOne() * this.cells.get(i).getDelO();
            this.derivativeLWrtRz += this.cells.get(i).getYtMinusOne() * this.cells.get(i).getDelZ();

            this.derivativeLWrtWi += this.cells.get(i).getXt() * this.cells.get(i).getDelI();
            this.derivativeLWrtWo += this.cells.get(i).getXt() * this.cells.get(i).getDelO();
            this.derivativeLWrtWz += this.cells.get(i).getXt() * this.cells.get(i).getDelZ();
        }
        
        // Average gradients? Original code divided by cells.size() in 'gradients' list construction.
        // Averaging is good for stability.
        double n = this.cells.size();
        var gradients = new ArrayList<Double>();
        gradients.add(this.derivativeLWrtWi / n);
        gradients.add(this.derivativeLWrtWo / n);
        gradients.add(this.derivativeLWrtWz / n);
        gradients.add(this.derivativeLWrtRi / n);
        gradients.add(this.derivativeLWrtRo / n);
        gradients.add(this.derivativeLWrtRz / n);

        this.updateweights(gradients);
    }

    protected void updateweights(ArrayList<Double> gradients) {
        var optimizer = new AdaptiveLearningRateImproved();

        // Update accumulated squared gradients
        for(int k=0; k<6; k++) {
            double g = gradients.get(k);
            double newG = this.accumulatedGradientSq.get(k) + g*g;
            this.accumulatedGradientSq.set(k, newG);
        }

        // Update weights for all cells (since they share weights)
        // Optimization: In standard LSTM, weights are matrices shared by all cells.
        // Here, each cell has its own copy of Wi, Wo etc. but they are set to be identical.
        // We calculate new weights once and apply to all.
        
        // Current weights (take from first cell)
        double curWi = this.cells.get(0).getWi();
        double curWo = this.cells.get(0).getWo();
        double curWz = this.cells.get(0).getWz();
        double curRi = this.cells.get(0).getRi();
        double curRo = this.cells.get(0).getRo();
        double curRz = this.cells.get(0).getRz();
        
        double newWi = optimizer.adagradUpdate(curWi, gradients.get(0), this.accumulatedGradientSq.get(0), learningRate, epsilon);
        double newWo = optimizer.adagradUpdate(curWo, gradients.get(1), this.accumulatedGradientSq.get(1), learningRate, epsilon);
        double newWz = optimizer.adagradUpdate(curWz, gradients.get(2), this.accumulatedGradientSq.get(2), learningRate, epsilon);
        double newRi = optimizer.adagradUpdate(curRi, gradients.get(3), this.accumulatedGradientSq.get(3), learningRate, epsilon);
        double newRo = optimizer.adagradUpdate(curRo, gradients.get(4), this.accumulatedGradientSq.get(4), learningRate, epsilon);
        double newRz = optimizer.adagradUpdate(curRz, gradients.get(5), this.accumulatedGradientSq.get(5), learningRate, epsilon);

        for (int i = 0; i < this.cells.size(); i++) {
            this.cells.get(i).setWi(newWi);
            this.cells.get(i).setWo(newWo);
            this.cells.get(i).setWz(newWz);
            this.cells.get(i).setRi(newRi);
            this.cells.get(i).setRo(newRo);
            this.cells.get(i).setRz(newRz);
        }
    }

    /**
     * Train to get the weight matrix.
     * 
     * NOTE: Original implementation found Global Minima based on training error in this method.
     * To support Validation-based selection, this method should return the history of weights
     * OR just train for N epochs and let the caller manage validation/checkpointing.
     * 
     * To keep signature compatible but improve logic:
     * We will return the weights at the END of training (last epoch).
     * The `TrainAndValidateBatch` class is responsible for calling this for a small number of iterations (gdIteration)
     * and checking validation error.
     * 
     * However, the original code called `model.fit()` which called `ls.train()`.
     * `ls.train()` runs `epoch` times.
     * `MakeModel.trainTrend` calls `model.fit`.
     * 
     * If we want to strictly follow "don't change old code behavior too much but improve correctness":
     * We should still return the best weights seen *during this training session*.
     * But relying on Training Error is risky.
     * 
     * Let's stick to the original behavior of returning the "Best on Training" for this specific method,
     * BUT ensuring the calculation is correct.
     * 
     * @return weight matrix trained weight matrix
     */
    public ArrayList<ArrayList<Double>> train() {
        var mW = new MatrixWeight();
        
        for (int i = 0; i < this.epoch; i++) {

            this.forwardprop(true);
            this.backwardprop();

            var wiList = new ArrayList<Double>();
            var woList = new ArrayList<Double>();
            var wzList = new ArrayList<Double>();
            var riList = new ArrayList<Double>();
            var roList = new ArrayList<Double>();
            var rzList = new ArrayList<Double>();
            var ytList = new ArrayList<Double>();
            var ctList = new ArrayList<Double>();

            for (int j = 0; j < this.cells.size(); j++) {
                wiList.add(this.cells.get(j).getWi());
                woList.add(this.cells.get(j).getWo());
                wzList.add(this.cells.get(j).getWz());
                riList.add(this.cells.get(j).getRi());
                roList.add(this.cells.get(j).getRo());
                rzList.add(this.cells.get(j).getRz());
                ytList.add(this.cells.get(j).getYt());
                ctList.add(this.cells.get(j).getCt());
            }

            mW.getErrorList().add(this.cells.get(this.cells.size() - 1).getError());
            mW.getWi().add(wiList);
            mW.getWo().add(woList);
            mW.getWz().add(wzList);
            mW.getRi().add(riList);
            mW.getRo().add(roList);
            mW.getRz().add(rzList);
            mW.getOut().add(ytList);
            mW.getCt().add(ctList);
        }

        // Original logic: Find best epoch based on training error
        int globalMinimaIndex = findGlobalMinima(mW.getErrorList());

        var returnArray = new ArrayList<ArrayList<Double>>();

        returnArray.add(mW.getWi().get(globalMinimaIndex));
        returnArray.add(mW.getWo().get(globalMinimaIndex));
        returnArray.add(mW.getWz().get(globalMinimaIndex));
        returnArray.add(mW.getRi().get(globalMinimaIndex));
        returnArray.add(mW.getRo().get(globalMinimaIndex));
        returnArray.add(mW.getRz().get(globalMinimaIndex));
        returnArray.add(mW.getOut().get(globalMinimaIndex));
        returnArray.add(mW.getCt().get(globalMinimaIndex));

        return returnArray;
    }

    public static int findGlobalMinima(ArrayList<Double> data) {
        return IntStream.range(0, data.size())
                .boxed()
                .min(Comparator.comparingDouble(i -> Math.abs(data.get(i))))
                .orElse(-1);
    }

    // Standard Setters/Getters
    public synchronized void setWi(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(0).size(); i++) {
            try {
                this.cells.get(i).setWi(val.get(0).get(i));
            } catch (Exception e) { e.printStackTrace(); }
        }
    }
    public synchronized void setWo(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(1).size(); i++) { this.cells.get(i).setWo(val.get(1).get(i)); }
    }
    public synchronized void setWz(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(2).size(); i++) { this.cells.get(i).setWz(val.get(2).get(i)); }
    }
    public synchronized void setRi(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(3).size(); i++) { this.cells.get(i).setRi(val.get(3).get(i)); }
    }
    public synchronized void setRo(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(4).size(); i++) { this.cells.get(i).setRo(val.get(4).get(i)); }
    }
    public synchronized void setRz(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(5).size(); i++) { this.cells.get(i).setRz(val.get(5).get(i)); }
    }
    public synchronized void setYt(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(6).size(); i++) { this.cells.get(i).setYt(val.get(6).get(i)); }
    }
    public synchronized void setCt(ArrayList<ArrayList<Double>> val) {
        for (int i = 0; i < val.get(7).size(); i++) { this.cells.get(i).setCt(val.get(7).get(i)); }
    }

    /**
     * Initializes the cell with the default data.
     */
    public synchronized void initilizeCells() {
        this.cells = new ArrayList<>();
        for (int i = 0; i < this.inputData.length; i++) {
            CellImproved cell = new CellImproved(this.inputData[i], this.outputData);
            this.cells.add(cell);
        }
    }

    public static class LstmBuilder {

        protected double[] inputData;
        protected double outputData;

        protected double learningRate;
        protected int epoch = 100;

        public LstmBuilder(double[] inputData, double outputData) {
            this.inputData = inputData;
            this.outputData = outputData;
        }

        public LstmBuilder setInputData(double[] inputData) {
            this.inputData = inputData;
            return this;
        }

        public LstmBuilder setOutputData(double outputData) {
            this.outputData = outputData;
            return this;
        }

        public LstmBuilder setLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public LstmBuilder setEpoch(int epoch) {
            this.epoch = epoch;
            return this;
        }

        public LstmImproved build() {
            return new LstmImproved(this);
        }
    }
}
