package com.littlemonstor.singlenodealg.utils;

import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.TwoClassStats;
import weka.core.*;

import java.util.ArrayList;
import java.util.Random;

public class MyThresholdCurve implements RevisionHandler {
    public static final String RELATION_NAME = "ThresholdCurve";
    public static final String TRUE_POS_NAME = "True Positives";
    public static final String FALSE_NEG_NAME = "False Negatives";
    public static final String FALSE_POS_NAME = "False Positives";
    public static final String TRUE_NEG_NAME = "True Negatives";
    public static final String FP_RATE_NAME = "False Positive Rate";
    public static final String TP_RATE_NAME = "True Positive Rate";
    public static final String PRECISION_NAME = "Precision";
    public static final String RECALL_NAME = "Recall";
    public static final String FALLOUT_NAME = "Fallout";
    public static final String FMEASURE_NAME = "FMeasure";
    public static final String SAMPLE_SIZE_NAME = "Sample Size";
    public static final String LIFT_NAME = "Lift";
    public static final String THRESHOLD_NAME = "Threshold";

    public MyThresholdCurve() {
    }

    public Instances getCurve(ArrayList<Prediction> predictions) {
        return predictions.size() == 0 ? null : this.getCurve(predictions, ((NominalPrediction)predictions.get(0)).distribution().length - 1);
    }

    public Instances getCurve(ArrayList<Prediction> predictions, int classIndex) {
        if (predictions.size() != 0 && ((NominalPrediction)predictions.get(0)).distribution().length > classIndex) {
            double totPos = 0.0D;
            double totNeg = 0.0D;
            double[] probs = this.getProbabilities(predictions, classIndex);

            for(int i = 0; i < probs.length; ++i) {
                NominalPrediction pred = (NominalPrediction)predictions.get(i);
                if (pred.actual() == Prediction.MISSING_VALUE) {
                    System.err.println(this.getClass().getName() + " Skipping prediction with missing class value");
                } else if (pred.weight() < 0.0D) {
                    System.err.println(this.getClass().getName() + " Skipping prediction with negative weight");
                } else if (pred.actual() == (double)classIndex) {
                    totPos += pred.weight();
                } else {
                    totNeg += pred.weight();
                }
            }

            Instances insts = this.makeHeader();
            int[] sorted = Utils.sort(probs);
            TwoClassStats tc = new TwoClassStats(totPos, totNeg, 0.0D, 0.0D);
            double threshold = 0.0D;
            double cumulativePos = 0.0D;
            double cumulativeNeg = 0.0D;

            for(int i = 0; i < sorted.length; ++i) {
                if (i == 0 || probs[sorted[i]] > threshold) {
                    tc.setTruePositive(tc.getTruePositive() - cumulativePos);
                    tc.setFalseNegative(tc.getFalseNegative() + cumulativePos);
                    tc.setFalsePositive(tc.getFalsePositive() - cumulativeNeg);
                    tc.setTrueNegative(tc.getTrueNegative() + cumulativeNeg);
                    threshold = probs[sorted[i]];
                    insts.add(this.makeInstance(tc, threshold));
                    cumulativePos = 0.0D;
                    cumulativeNeg = 0.0D;
                    if (i == sorted.length - 1) {
                        break;
                    }
                }

                NominalPrediction pred = (NominalPrediction)predictions.get(sorted[i]);
                if (pred.actual() == Prediction.MISSING_VALUE) {
                    System.err.println(this.getClass().getName() + " Skipping prediction with missing class value");
                } else if (pred.weight() < 0.0D) {
                    System.err.println(this.getClass().getName() + " Skipping prediction with negative weight");
                } else if (pred.actual() == (double)classIndex) {
                    cumulativePos += pred.weight();
                } else {
                    cumulativeNeg += pred.weight();
                }
            }

            if (tc.getFalseNegative() != totPos || tc.getTrueNegative() != totNeg) {
                tc = new TwoClassStats(0.0D, 0.0D, totNeg, totPos);
                threshold = probs[sorted[sorted.length - 1]] + 1.0E-5D;
                insts.add(this.makeInstance(tc, threshold));
            }

            return insts;
        } else {
            return null;
        }
    }

    public static double getNPointPrecision(Instances tcurve, int n) {
        if ("ThresholdCurve".equals(tcurve.relationName()) && tcurve.numInstances() != 0) {
            int recallInd = tcurve.attribute("Recall").index();
            int precisInd = tcurve.attribute("Precision").index();
            double[] recallVals = tcurve.attributeToDoubleArray(recallInd);
            int[] sorted = Utils.sort(recallVals);
            double isize = 1.0D / (double)(n - 1);
            double psum = 0.0D;

            for(int i = 0; i < n; ++i) {
                int pos = binarySearch(sorted, recallVals, (double)i * isize);
                double recall = recallVals[sorted[pos]];
                double precis = tcurve.instance(sorted[pos]).value(precisInd);

                while(pos != 0 && pos < sorted.length - 1) {
                    ++pos;
                    double recall2 = recallVals[sorted[pos]];
                    if (recall2 != recall) {
                        double precis2 = tcurve.instance(sorted[pos]).value(precisInd);
                        double slope = (precis2 - precis) / (recall2 - recall);
                        double offset = precis - recall * slope;
                        precis = isize * (double)i * slope + offset;
                        break;
                    }
                }

                psum += precis;
            }

            return psum / (double)n;
        } else {
            return 0.0D / 0.0;
        }
    }

    public static double getPRCArea(Instances tcurve) {
        int n = tcurve.numInstances();
        if ("ThresholdCurve".equals(tcurve.relationName()) && n != 0) {
            int pInd = tcurve.attribute("Precision").index();
            int rInd = tcurve.attribute("Recall").index();
            double[] pVals = tcurve.attributeToDoubleArray(pInd);
            double[] rVals = tcurve.attributeToDoubleArray(rInd);
            double area = 0.0D;
            double xlast = rVals[n - 1];

            for(int i = n - 2; i >= 0; --i) {
                double recallDelta = rVals[i] - xlast;
                area += pVals[i] * recallDelta;
                xlast = rVals[i];
            }

            return area == 0.0D ? Utils.missingValue() : area;
        } else {
            return 0.0D / 0.0;
        }
    }

    public static double getROCArea(Instances tcurve) {
        int n = tcurve.numInstances();
        if ("ThresholdCurve".equals(tcurve.relationName()) && n != 0) {
            int tpInd = tcurve.attribute("True Positives").index();
            int fpInd = tcurve.attribute("False Positives").index();
            double[] tpVals = tcurve.attributeToDoubleArray(tpInd);
            double[] fpVals = tcurve.attributeToDoubleArray(fpInd);
            double area = 0.0D;
            double cumNeg = 0.0D;
            double totalPos = tpVals[0];
            double totalNeg = fpVals[0];
            if(totalNeg==0||totalPos==0){
                return -1.0;
            }
            for(int i = 0; i < n; ++i) {
                double cip;
                double cin;
                if (i < n - 1) {
                    cip = tpVals[i] - tpVals[i + 1];
                    cin = fpVals[i] - fpVals[i + 1];
                } else {
                    cip = tpVals[n - 1];
                    cin = fpVals[n - 1];
                }

                area += cip * (cumNeg + 0.5D * cin);
                cumNeg += cin;
            }

            area /= totalNeg * totalPos;
            return area;
        } else {
            return 0.0D / 0.0;
        }
    }

    public static int getThresholdInstance(Instances tcurve, double threshold) {
        if ("ThresholdCurve".equals(tcurve.relationName()) && tcurve.numInstances() != 0 && threshold >= 0.0D && threshold <= 1.0D) {
            if (tcurve.numInstances() == 1) {
                return 0;
            } else {
                double[] tvals = tcurve.attributeToDoubleArray(tcurve.numAttributes() - 1);
                int[] sorted = Utils.sort(tvals);
                return binarySearch(sorted, tvals, threshold);
            }
        } else {
            return -1;
        }
    }

    private static int binarySearch(int[] index, double[] vals, double target) {
        int lo = 0;
        int hi = index.length - 1;

        while(hi - lo > 1) {
            int mid = lo + (hi - lo) / 2;
            double midval = vals[index[mid]];
            if (target > midval) {
                lo = mid;
            } else {
                if (target >= midval) {
                    while(mid > 0 && vals[index[mid - 1]] == target) {
                        --mid;
                    }

                    return mid;
                }

                hi = mid;
            }
        }

        return lo;
    }

    private double[] getProbabilities(ArrayList<Prediction> predictions, int classIndex) {
        double[] probs = new double[predictions.size()];

        for(int i = 0; i < probs.length; ++i) {
            NominalPrediction pred = (NominalPrediction)predictions.get(i);
            probs[i] = pred.distribution()[classIndex];
        }

        return probs;
    }

    private Instances makeHeader() {
        ArrayList<Attribute> fv = new ArrayList();
        fv.add(new Attribute("True Positives"));
        fv.add(new Attribute("False Negatives"));
        fv.add(new Attribute("False Positives"));
        fv.add(new Attribute("True Negatives"));
        fv.add(new Attribute("False Positive Rate"));
        fv.add(new Attribute("True Positive Rate"));
        fv.add(new Attribute("Precision"));
        fv.add(new Attribute("Recall"));
        fv.add(new Attribute("Fallout"));
        fv.add(new Attribute("FMeasure"));
        fv.add(new Attribute("Sample Size"));
        fv.add(new Attribute("Lift"));
        fv.add(new Attribute("Threshold"));
        return new Instances("ThresholdCurve", fv, 100);
    }

    private Instance makeInstance(TwoClassStats tc, double prob) {
        int count = 0;
        double[] vals = new double[13];
        int var10 = count + 1;
        vals[count] = tc.getTruePositive();
        vals[var10++] = tc.getFalseNegative();
        vals[var10++] = tc.getFalsePositive();
        vals[var10++] = tc.getTrueNegative();
        vals[var10++] = tc.getFalsePositiveRate();
        vals[var10++] = tc.getTruePositiveRate();
        vals[var10++] = tc.getPrecision();
        vals[var10++] = tc.getRecall();
        vals[var10++] = tc.getFallout();
        vals[var10++] = tc.getFMeasure();
        double ss = (tc.getTruePositive() + tc.getFalsePositive()) / (tc.getTruePositive() + tc.getFalsePositive() + tc.getTrueNegative() + tc.getFalseNegative());
        vals[var10++] = ss;
        double expectedByChance = ss * (tc.getTruePositive() + tc.getFalseNegative());
        if (expectedByChance < 1.0D) {
            vals[var10++] = Utils.missingValue();
        } else {
            vals[var10++] = tc.getTruePositive() / expectedByChance;
        }

        vals[var10++] = prob;
        return new DenseInstance(1.0D, vals);
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10153 $");
    }

    public static void main(String[] args) {
        try {
            MyThresholdCurve m=new MyThresholdCurve();
            ArrayList<Prediction> pl=new ArrayList<>();
            Random r=new Random();
            for(int i=0;i<100;i++){
                double x=r.nextDouble();
                double rx=r.nextDouble();
                double lable=0;
                if(rx>0.5){
                    lable=1;
                }
                System.out.println(lable+","+x);

                pl.add(new NominalPrediction(lable,new double[]{x,1-x}));
            }
            ;
            System.out.println(MyThresholdCurve.getROCArea(m.getCurve(pl)));
        } catch (Exception var7) {
            var7.printStackTrace();
        }

    }
}
