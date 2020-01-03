package com.littlemonstor.singlenodealg.linear;

import com.alibaba.fastjson.JSON;

import com.littlemonstor.singlenodealg.utils.ModelAndByteUtils;
import com.littlemonstor.singlenodealg.utils.MyThresholdCurve;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.*;

public class FTRL implements Serializable {
    private final static Logger logger = LoggerFactory.getLogger(FTRL.class);
    private static final long serialVersionUID = -457725757758289986L;
    private double alpha=1;
    private double beta=1;
    private double l1=1;
    private double l2=1;
    private double subsamplePositive=1;
    private double subsampleNegative=1;
    private int epochs=1;

    private double logLikelyhood=0;

    private double[] z=null;
    private double[] n=null;
    private double targetRatio=0;

    private boolean fitFlag=false;
    private int featureLen=0;
    private int sampleNum=0;
    private int trainPrintF=100000;
    private int testPrintF=2000000;
    private String testSamplePath;
    private String trainPath;
    private String savePath;
    private String monitorPath;
    private int trainSampleNum;
    private double logLoss(double y,double p){
        p=Math.max(Math.min(p, 1. - 10e-15),10e-15);
        return y==1?Math.log(p):Math.log(1.-p);
    }

    public FTRL(){

    }
    public FTRL(Map<String,Object> params){
        if(params.get("alpha")!=null){
            this.alpha=(Double) params.get("alpha");
        }
        if(params.get("beta")!=null){
            this.beta=(Double)params.get("beta");
        }
        if(params.get("l1")!=null){
            this.l1=(Double)params.get("l1");
        }
        if(params.get("l2")!=null){
            this.l2=(Double)params.get("l2");
        }
        if(params.get("subSamplePos")!=null){
            this.subsamplePositive=(Double)params.get("subSamplePos");
        }
        if(params.get("subSampleNeg")!=null){
            this.subsampleNegative=(Double)params.get("subSampleNeg");
        }
        if(params.get("epochs")!=null){
            this.epochs=(Integer) params.get("epochs");
        }
        if(params.get("trainPrintF")!=null){
            this.trainPrintF=(Integer) params.get("trainPrintF");
        }
        if(params.get("testPrintF")!=null){
            this.testPrintF=(Integer) params.get("testPrintF");
        }
        if(params.get("testPath")!=null){
            this.testSamplePath=(String)params.get("testPath");
        }
        if(params.get("trainPath")!=null){
            this.trainPath=(String)params.get("trainPath");
        }
        if(params.get("savePath")!=null){
            this.savePath=(String)params.get("savePath");
        }
        if(params.get("sampleNum")!=null){
            this.sampleNum=(Integer)params.get("sampleNum");
        }
        if(params.get("monitorPath")!=null){
            this.monitorPath=(String)params.get("monitorPath");
        }


        System.out.println("=====params===="+JSON.toJSONString(params));


    }

    private void update(double y ,double p, int[] x,double[] w){
        for(int i=0;i<x.length;i++){
            int oneIndex=x[i];
            double g=(p-y);
            double s=(Math.sqrt(this.n[oneIndex]+g*g)-Math.sqrt(this.n[oneIndex]))/this.alpha;
            this.z[oneIndex]+=g-s*w[i];
            this.n[oneIndex]+=g*g;
        }
    }
    public void scratch(int startIndex) throws Exception{
        LineIterator it =null;

        try {
             it = FileUtils.lineIterator(new File(this.trainPath)
                    , "UTF-8");
            double positiveRatio = 0;
            double sampleTrainCount = 0;
            double positiveSample = 0;
            int maxDim = 0;
            SimpleDateFormat s = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            while (it.hasNext()) {

                String line = it.nextLine();
                String[] elems = line.split(",");

                int y = Integer.valueOf(elems[startIndex]);
                positiveSample = positiveSample+y;
                positiveRatio = (1.0 * (sampleTrainCount * positiveRatio + y)) / (sampleTrainCount + 1);


                int[] x = new int[elems.length - startIndex];
                for (int i = startIndex + 1; i < elems.length; i++) {
                    maxDim = Math.max(Integer.valueOf(elems[i]), maxDim);

                }

                sampleTrainCount++;

            }

            Date d = new Date();
            this.featureLen =maxDim+1;
            logger.info(s.format(d) + "---sample=" + sampleTrainCount + ",positiveRatio="
                    + positiveRatio + ", feature max dim=" + maxDim);


        }finally {
            LineIterator.closeQuietly(it);
        }
        this.z=new double[featureLen];
        this.n=new double[featureLen];
        for(int index=0;index<featureLen;index++){
            this.z[index]=0;
            this.n[index]=0;
        }


    }
    public void train(int startIndex){
        LineIterator it=null;
        Random r=new Random();
        Date start=new Date();
        int sampleTrainCount=0;
        int runEpoch=0;
        Map<String, ArrayList<Prediction>> userPl= new HashMap<>();
        ArrayList<Prediction> pl=new ArrayList<>();
        while(this.epochs>runEpoch){
            try {

                it = FileUtils.lineIterator(new File(this.trainPath)
                        , "UTF-8");
                while (it.hasNext() ) {
                    if(sampleTrainCount>this.sampleNum){
                        break;
                    }
                    String line = it.nextLine();
                    String[] elems=line.split(",");
                    if(startIndex>=elems.length) {
                        continue;
                    }
                    int y=Integer.valueOf(elems[startIndex]);

                    double nextR=r.nextDouble();
                    if(y<1){
                        if(nextR>this.subsampleNegative ){
                            continue;
                        }
                    }else{
                        if(nextR>this.subsamplePositive ){
                            continue;
                        }
                    }
                    sampleTrainCount+=1;
                    int[] x=new int[elems.length-startIndex];
                    for(int i=startIndex+1;i<elems.length;i++){
                        x[i-startIndex-1]=Integer.valueOf(elems[i]);
                    }


                    this.targetRatio=(1.0*(sampleTrainCount*targetRatio+y))/(sampleTrainCount+1);

                    double wx=0;
                    double[] w=new double[x.length];
                    for(int i=0;i<x.length;i++){
                        if(Math.abs(z[x[i]])<=this.l1){
                            w[i]=0;
                        }else{
                            int sign=this.z[x[i]]>=0?1:-1;
                            w[i]= -(this.z[x[i]]-sign*this.l1)/(this.l2+(this.beta+Math.sqrt(this.n[x[i]]))/this.alpha);

                        }
                        wx+=w[i];
                    }
                    double p=1. / (1. + Math.exp(-Math.max(Math.min(wx, 35.), -35.)));
                    pl.add(new NominalPrediction(y,new double[]{1-p,p}));
                    ArrayList<Prediction> pi = userPl.get(elems[0]);
                    if (pi == null) {
                        pi = new ArrayList<Prediction>();
                        userPl.put(elems[0],pi);
                    }
                    pi.add(new NominalPrediction(y,new double[]{1-p,p}));
                    this.logLikelyhood += logLoss(y, p);
                    if(sampleTrainCount%this.trainPrintF==0){
                        SimpleDateFormat s=new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                        Date d=new Date();
                        double avgLoss=this.logLikelyhood/sampleTrainCount;
                        double finishRatio=1.0*sampleTrainCount/sampleNum;
                        String finishRatioStr=String.format("%.5f", finishRatio);
                        String avgLossStr=String.format("%.5f", avgLoss);
                        logger.info(s.format(d)+"---train sample="+sampleTrainCount+",positiveRatio="
                                +targetRatio+"---- loss="+avgLossStr+"---train time="+((d.getTime()-start.getTime())/1000)+",finish ratio="+
                                 finishRatioStr+"---runEpoch="+runEpoch);
                        MyThresholdCurve m=new MyThresholdCurve();
                        logger.info( "AUC====="+MyThresholdCurve.getROCArea(m.getCurve(pl)));


                        pl.clear();

                    }
                    if(sampleTrainCount%this.testPrintF==0){
                        test(2);
                    }

                    this.update(y,p,x,w);

                }
            }catch (Exception e){
                e.printStackTrace();
            }finally {
                LineIterator.closeQuietly(it);
            }
            runEpoch++;
        }

    }

    public double predict(int[] x){
        double wx=0;
        double[] w=new double[x.length];
        try{
            for(int i=0;i<x.length;i++){
                if(z.length<=x[i]||n.length<=x[i]){
                    continue;

                }
                if(Math.abs(z[x[i]])<=this.l1){
                    w[i]=0;
                }else{
                    int sign=this.z[x[i]]>=0?1:-1;
                    w[i]= -(this.z[x[i]]-sign*this.l1)/(this.l2+(this.beta+Math.sqrt(this.n[x[i]]))/this.alpha);

                }
                wx+=w[i];
            }
            double p=1. / (1. + Math.exp(-Math.max(Math.min(wx, 35.), -35.)));
            return p;
        }catch(Exception e){
            return -1;
        }
    }
    public Map<Integer,Double> getW(int[] x){
        Map<Integer,Double> w=new HashMap<>();
        for(int i=0;i<x.length;i++){
            if(z.length<=x[i]||n.length<=x[i]){
                continue;
            }
            if(Math.abs(z[x[i]])<=this.l1){
                w.put(x[i],0.0);
            }else{
                int sign=this.z[x[i]]>=0?1:-1;
                double wi =-(this.z[x[i]]-sign*this.l1)/(this.l2+(this.beta+Math.sqrt(this.n[x[i]]))/this.alpha);
                w.put(x[i],wi);
            }
        }
        return w;
    }
    public ArrayList<String> test(int startIndex){
        LineIterator it =null;
        ArrayList<String> pls=new ArrayList<>();
        try {
            double loss =0d;
            int count=0;
            ArrayList<Prediction> pl=new ArrayList<>();
            Map<String, ArrayList<Prediction>> userPl= new HashMap<>();
            it = FileUtils.lineIterator(new File(this.testSamplePath)
                    , "UTF-8");
            while (it.hasNext() ) {
                String line = it.nextLine();
                String[] elems=line.split(",");
                if(startIndex>=elems.length){
                    continue;
                }
                int y=Integer.valueOf(elems[startIndex]);
                int[] x=new int[elems.length-startIndex];
                for(int i=startIndex+1;i<elems.length;i++){
                    x[i-startIndex-1]=Integer.valueOf(elems[i]);
                }
                double p=predict(x);
                pls.add(y+"\t"+p);
                loss += logLoss(y, p);
                pl.add(new NominalPrediction(y,new double[]{1-p,p}));
                ArrayList<Prediction> pi = userPl.get(elems[0]);
                if (pi == null) {
                    pi = new ArrayList<Prediction>();
                    userPl.put(elems[0],pi);
                }
                pi.add(new NominalPrediction(y,new double[]{1-p,p}));
                count++;
            }
            MyThresholdCurve m=new MyThresholdCurve();
            logger.info( "TEST SET AUC====="+MyThresholdCurve.getROCArea(m.getCurve(pl))+"===" +
                    "TEST SET LOSS="+(loss/count));

            pl.clear();


        } catch (IOException e) {
            e.printStackTrace();
        }
        return pls;
    }
    public void write2File()throws Exception{
        FileUtils.writeByteArrayToFile(new File(this.savePath), ModelAndByteUtils.object2byte(this));

    }
    public FTRL loadFromFile(String f)throws Exception{
        return (FTRL)ModelAndByteUtils.byte2Object(FileUtils.readFileToByteArray(new File(f)));
    }
    public static void main(String[] args)throws Exception{
        Map<String,Object> param=new HashMap<>();
        param.put("alpha",Double.valueOf(args[0]));
        param.put("beta",Double.valueOf(args[1]));
        param.put("l1",Double.valueOf(args[2]));
        param.put("l2",Double.valueOf(args[3]));
        param.put("subSamplePos",Double.valueOf(args[4]));
        param.put("subSampleNeg",Double.valueOf(args[5]));
        param.put("epochs",Integer.valueOf(args[6]));
        param.put("sampleNum",Integer.valueOf(args[7]));
        param.put("trainPrintF",Integer.valueOf(args[8]));
        param.put("testPrintF",Integer.valueOf(args[9]));
        param.put("trainPath",args[10]);
        param.put("testPath",args[11]);
        param.put("savePath",args[12]);
        param.put("monitorPath",args[13]);

        FTRL ftrl=new FTRL(param);
        ftrl.scratch(2);
        ftrl.train(2);
        ftrl.write2File();


    }
}
