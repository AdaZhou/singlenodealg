package com.littlemonstor.singlenodealg.utils;

import java.io.*;

public class ModelAndByteUtils {
    public static byte[] object2byte(Object encoder) throws Exception {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        try {
            ObjectOutputStream oos = new ObjectOutputStream(byteArrayOutputStream);

            oos.writeObject(encoder);
            byte[] value = byteArrayOutputStream.toByteArray();
            return value;
        } catch (IOException e) {
            throw e;
        }

    }

    public static Object byte2Object(byte[] bytes) throws Exception {
        ByteArrayInputStream bais = null;

        try {


            bais = new ByteArrayInputStream(bytes);

            ObjectInputStream ois = new ObjectInputStream(bais);

            return  ois.readObject();

        } catch (Exception e) {
            throw e;


        }


    }
}
