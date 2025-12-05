package predictor.lstm.util.improved;

import java.util.Random;
import predictor.lstm.utilities.MathUtils;

public class CellImproved {

    private double error;
    private double wI;
    private double wO;
    private double wZ;

    private double rI;
    private double rO;
    private double rZ;

    private double yT;
    private double ytMinusOne;

    private double cT;
    private double ctMinusOne;
    private double oT;
    private double zT;

    private double iT;
    private double dlByDy;
    private double dlByDo;
    private double dlByDc;
    private double dlByDi;
    private double dlByDz;
    private double delI;
    private double delO;
    private double delZ;

    private double xT;
    private double outputDataLoc;

    // Dropout fields
    private boolean dropoutEnabled = false;
    private double dropoutRate = 0.0;
    private double dropoutMask = 1.0; // 1.0 means keep, 0.0 means drop

    public CellImproved(double xt, double outputData) {
        this(xt, outputData, 1, 1, 1, 1, 1, 1, 0);
    }

    public CellImproved(double xt, double outputData, double wI, double wO, double wZ, double rI, double rO, double rZ,
                double yT) {
        this.dlByDc = 0;
        this.error = 0;
        this.wI = wI;
        this.wO = wO;
        this.wZ = wZ;
        this.rI = rI;
        this.rO = rO;
        this.rZ = rZ;
        this.cT = 0;
        this.oT = 0;
        this.zT = 0;
        this.yT = 0;
        this.ytMinusOne = 0;
        this.ctMinusOne = 0;
        this.ytMinusOne = this.yT;
        this.dlByDy = 0;
        this.dlByDo = 0;
        this.dlByDc = 0;
        this.dlByDi = 0;
        this.dlByDz = 0;
        this.delI = 0;
        this.delO = 0;
        this.delZ = 0;
        this.iT = 0;
        this.xT = xt;
        this.outputDataLoc = outputData;
    }

    public void setDropoutEnabled(boolean enabled) {
        this.dropoutEnabled = enabled;
    }

    public void setDropoutRate(double rate) {
        this.dropoutRate = rate;
    }

    /**
     * Forward propagation with standard Inverted Dropout.
     */
    public void forwardPropogation() {
        this.iT = MathUtils.sigmoid(this.wI * this.xT + this.rI * this.ytMinusOne);
        this.oT = MathUtils.sigmoid(this.wO * this.xT + this.rO * this.ytMinusOne);
        this.zT = MathUtils.tanh(this.wZ * this.xT + this.rZ * this.ytMinusOne);
        
        // Calculate candidate cell state
        this.cT = this.ctMinusOne + this.iT * this.zT;
        
        // Calculate raw output
        double rawYt = this.oT * MathUtils.tanh(this.cT);

        // Apply Dropout
        if (this.dropoutEnabled && this.dropoutRate > 0) {
            // Inverted Dropout:
            // 1. Generate mask (1 with probability (1-p), 0 with probability p)
            // 2. Scale by 1/(1-p) during training so no scaling is needed during testing
            
            Random random = new Random();
            boolean keep = random.nextDouble() >= this.dropoutRate;
            this.dropoutMask = keep ? (1.0 / (1.0 - this.dropoutRate)) : 0.0;
            
            this.yT = rawYt * this.dropoutMask;
        } else {
            // During testing or no dropout, use raw output
            // Since we scaled during training, we don't scale here.
            this.yT = rawYt;
            this.dropoutMask = 1.0; // Implicitly kept
        }

        this.error = this.yT - this.outputDataLoc;
    }

    /**
     * Backward propagation.
     */
    public void backwardPropogation() {
        // Gradient of Loss w.r.t Output yT
        // Assuming MSE Loss: L = 1/2 * (yT - target)^2
        // dL/dyT = (yT - target) = error
        // Note: The original code had signum(error) / sqrt(2), which was weird.
        this.dlByDy = this.error;

        // If dropout was applied, we must pass gradient through the mask
        if (this.dropoutEnabled && this.dropoutRate > 0) {
             this.dlByDy = this.dlByDy * this.dropoutMask;
        }

        // Backprop through Output Gate (oT) and Cell State (cT)
        // yT = oT * tanh(cT)
        // dl/doT = dl/dyT * tanh(cT)
        this.dlByDo = this.dlByDy * MathUtils.tanh(this.cT);
        
        // dl/dcT (current step contribution) = dl/dyT * oT * (1 - tanh^2(cT))
        double dlByDcCurrent = this.dlByDy * this.oT * MathUtils.tanhDerivative(this.cT);
        
        // Add gradient from next time step (dlByDc accumulated from future)
        this.dlByDc = dlByDcCurrent + this.dlByDc;

        // Backprop through Input Gate (iT) and Candidate Cell State (zT)
        // cT = ctMinusOne + iT * zT
        
        // dl/dzT = dl/dcT * iT
        this.dlByDz = this.dlByDc * this.iT;
        
        // dl/diT = dl/dcT * zT
        this.dlByDi = this.dlByDc * this.zT;

        // Gradients for gates (activations)
        // iT = sigmoid(...) -> dl/dInput = dl/diT * iT * (1 - iT)
        this.delI = this.dlByDi * MathUtils.sigmoidDerivative(this.wI * this.xT + this.rI * this.ytMinusOne);
        
        // oT = sigmoid(...) -> dl/dInput = dl/doT * oT * (1 - oT)
        this.delO = this.dlByDo * MathUtils.sigmoidDerivative(this.wO * this.xT + this.rO * this.ytMinusOne);
        
        // zT = tanh(...) -> dl/dInput = dl/dzT * (1 - zT^2)
        this.delZ = this.dlByDz * MathUtils.tanhDerivative(this.wZ * this.xT + this.rZ * this.ytMinusOne);
        
        // Note: Gradients w.r.t weights (Wi, Ri, etc.) are calculated in LstmImproved class
        // by accumulating delI, delO, delZ multiplied by inputs (xT, ytMinusOne).
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

    public double getRi() { return this.rI; }
    public void setRi(double ri) { this.rI = ri; }

    public double getRo() { return this.rO; }
    public void setRo(double ro) { this.rO = ro; }

    public double getRz() { return this.rZ; }
    public void setRz(double rz) { this.rZ = rz; }

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

    public double getXt() { return this.xT; }
    public void setXt(double xt) { this.xT = xt; }
}
