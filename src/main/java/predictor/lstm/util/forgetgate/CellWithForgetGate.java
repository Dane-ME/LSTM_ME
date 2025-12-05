package predictor.lstm.util.forgetgate;

import java.util.Random;
import predictor.lstm.utilities.MathUtils;

public class CellWithForgetGate {

    private double error;
    
    // Input Gate
    private double wI;
    private double rI;
    
    // Output Gate
    private double wO;
    private double rO;
    
    // Candidate Gate (Cell Input)
    private double wZ;
    private double rZ;
    
    // Forget Gate (NEW)
    private double wF;
    private double rF;

    private double yT;
    private double ytMinusOne;

    private double cT;
    private double ctMinusOne;
    
    // Gate activations
    private double iT;
    private double oT;
    private double zT;
    private double fT; // NEW

    // Gradients
    private double dlByDy;
    private double dlByDo;
    private double dlByDc;
    private double dlByDi;
    private double dlByDz;
    private double dlByDf; // NEW
    
    private double delI;
    private double delO;
    private double delZ;
    private double delF; // NEW

    private double xT;
    private double outputDataLoc;

    // Dropout fields
    private boolean dropoutEnabled = false;
    private double dropoutRate = 0.0;
    private double dropoutMask = 1.0; 

    public CellWithForgetGate(double xt, double outputData) {
        // Default init with 1s, but usually set explicitly later
        this.wI = 1; this.wO = 1; this.wZ = 1; this.wF = 1;
        this.rI = 1; this.rO = 1; this.rZ = 1; this.rF = 1;
        this.xT = xt;
        this.outputDataLoc = outputData;
        this.ctMinusOne = 0;
        this.ytMinusOne = 0;
    }

    public void setDropoutEnabled(boolean enabled) {
        this.dropoutEnabled = enabled;
    }

    public void setDropoutRate(double rate) {
        this.dropoutRate = rate;
    }

    /**
     * Forward propagation with Forget Gate.
     */
    public void forwardPropogation() {
        // Calculate gates
        this.iT = MathUtils.sigmoid(this.wI * this.xT + this.rI * this.ytMinusOne);
        this.oT = MathUtils.sigmoid(this.wO * this.xT + this.rO * this.ytMinusOne);
        this.zT = MathUtils.tanh(this.wZ * this.xT + this.rZ * this.ytMinusOne);
        this.fT = MathUtils.sigmoid(this.wF * this.xT + this.rF * this.ytMinusOne); // NEW
        
        // Calculate cell state: c_t = f_t * c_{t-1} + i_t * z_t
        this.cT = this.fT * this.ctMinusOne + this.iT * this.zT;
        
        // Calculate raw output
        double rawYt = this.oT * MathUtils.tanh(this.cT);

        // Apply Dropout (Inverted)
        if (this.dropoutEnabled && this.dropoutRate > 0) {
            Random random = new Random();
            boolean keep = random.nextDouble() >= this.dropoutRate;
            this.dropoutMask = keep ? (1.0 / (1.0 - this.dropoutRate)) : 0.0;
            this.yT = rawYt * this.dropoutMask;
        } else {
            this.yT = rawYt;
            this.dropoutMask = 1.0;
        }

        this.error = this.yT - this.outputDataLoc;
    }

    /**
     * Backward propagation with Forget Gate.
     */
    public void backwardPropogation() {
        // dL/dyT
        this.dlByDy = this.error;

        if (this.dropoutEnabled && this.dropoutRate > 0) {
             this.dlByDy = this.dlByDy * this.dropoutMask;
        }

        // dL/doT
        this.dlByDo = this.dlByDy * MathUtils.tanh(this.cT);
        
        // dL/dcT (current step contribution)
        // yT = oT * tanh(cT) -> dy/dc = oT * (1-tanh^2)
        double dlByDcCurrent = this.dlByDy * this.oT * MathUtils.tanhDerivative(this.cT);
        
        // Accumulate gradient from future (dlByDc passed from next cell)
        this.dlByDc = dlByDcCurrent + this.dlByDc;

        // Gates gradients from Cell State
        // cT = fT * ctMinusOne + iT * zT
        
        // dL/dzT = dL/dcT * iT
        this.dlByDz = this.dlByDc * this.iT;
        
        // dL/diT = dL/dcT * zT
        this.dlByDi = this.dlByDc * this.zT;
        
        // dL/dfT = dL/dcT * ctMinusOne
        this.dlByDf = this.dlByDc * this.ctMinusOne;

        // Gradient for previous cell state (to pass to previous cell)
        // dL/dctMinusOne = dL/dcT * fT
        // NOTE: This value effectively updates `dlByDc` of the *previous* cell in the LSTM loop.
        // We don't store it in `this.dlByDc`, but we calculate it implicitly when backpropagating through time.
        // Actually, in the LSTM loop, we do: cell[i-1].dlByDc = cell[i].dlByDc * fT ?
        // No, `cell[i-1].dlByDc` accumulates `dL/dc_{t-1}`.
        // `dL/dc_{t-1} = dL/dy_{t-1} * ... + dL/dc_t * df/dc_{t-1} + dL/dc_t * di/dc_{t-1} ...`
        // Wait, gates depend on h_{t-1} (y_{t-1}), NOT c_{t-1} directly (unless using peephole).
        // So `dL/dc_{t-1}` comes purely from `dL/dc_t * dc_t/dc_{t-1}`.
        // `dc_t/dc_{t-1} = fT`.
        // So `dL/dc_{t-1} += dL/dc_t * fT`. 
        // This logic is handled in the Lstm loop by passing `dlByDc * fT`.
        // Wait, the `dlByDc` field in THIS class represents `dL/dc_t`.
        // We need to expose `fT` so the LSTM loop can calculate `dL/dc_{t-1}`.

        // Activation gradients (sigmoid/tanh derivative)
        this.delI = this.dlByDi * MathUtils.sigmoidDerivative(this.wI * this.xT + this.rI * this.ytMinusOne);
        this.delO = this.dlByDo * MathUtils.sigmoidDerivative(this.wO * this.xT + this.rO * this.ytMinusOne);
        this.delZ = this.dlByDz * MathUtils.tanhDerivative(this.wZ * this.xT + this.rZ * this.ytMinusOne);
        this.delF = this.dlByDf * MathUtils.sigmoidDerivative(this.wF * this.xT + this.rF * this.ytMinusOne);
    }

    // Getters and Setters
    public double getError() { return this.error; }
    public void setError(double error) { this.error = error; }

    public double getWi() { return this.wI; }
    public void setWi(double wi) { this.wI = wi; }

    public double getWo() { return this.wO; }
    public void setWo(double wo) { this.wO = wo; }

    public double getWz() { return this.wZ; }
    public void setWz(double wz) { this.wZ = wz; }
    
    public double getWf() { return this.wF; }
    public void setWf(double wf) { this.wF = wf; }

    public double getRi() { return this.rI; }
    public void setRi(double ri) { this.rI = ri; }

    public double getRo() { return this.rO; }
    public void setRo(double ro) { this.rO = ro; }

    public double getRz() { return this.rZ; }
    public void setRz(double rz) { this.rZ = rz; }
    
    public double getRf() { return this.rF; }
    public void setRf(double rf) { this.rF = rf; }

    public double getCt() { return this.cT; }
    public void setCt(double ct) { this.cT = ct; }

    public double getCtMinusOne() { return this.ctMinusOne; }
    public void setCtMinusOne(double ct) { this.ctMinusOne = ct; }

    public double getYtMinusOne() { return this.ytMinusOne; }
    public void setYtMinusOne(double yt) { this.ytMinusOne = yt; }

    public double getYt() { return this.yT; }
    public void setYt(double yt) { this.yT = yt; }

    public double getDlByDc() { return this.dlByDc; }
    public void setDlByDc(double dlByDc) { this.dlByDc = dlByDc; }

    public double getDelI() { return this.delI; }
    public double getDelO() { return this.delO; }
    public double getDelZ() { return this.delZ; }
    public double getDelF() { return this.delF; }
    public double getFt() { return this.fT; } // Needed for BPTT

    public double getXt() { return this.xT; }
    public void setXt(double xt) { this.xT = xt; }
}
