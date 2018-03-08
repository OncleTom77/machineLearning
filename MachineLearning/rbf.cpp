//
//  rbf.cpp
//  MachineLearning
//
//  Created by Thomas Fouan on 08/03/2018.
//  Copyright © 2018 Thomas Fouan. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include <math.h>

extern "C" {
    
    double getDistance(double* first, double* second, int length) {
        double result = 0.0;
        
        for (int i = 0; i < length; i++) {
            result += pow(first[i] - second[i], 2);
        }
        
        return sqrt(result);
    }
    
    double* rbf_create(double* inputs, int nbParameters, double* outputs, int nbExamples, double gamma) {
        double* w = (double*) malloc(sizeof(double) * nbParameters);
        int i, j;
        Eigen::MatrixXd x(nbExamples, nbExamples);
        Eigen::MatrixXd y(nbExamples, 1);
        
        for (i = 0; i < nbExamples; i++) {
            for (j = 0; j < nbExamples; j++) {
                double distance = getDistance(inputs + (i * nbParameters), inputs + (j * nbParameters), nbParameters);
                x(i, j) = exp(-gamma * abs(distance));
            }
            
            y(i, 0) = outputs[i];
        }
        
        Eigen::MatrixXd matrixW = x.inverse() * y;
        
        for (i = 0; i < nbExamples; i++) {
            w[i] = matrixW(i, 0);
        }
        
        return w;
    }
    
    double get_result_from_rbf(double* w, double* inputs, int nbParameters, int nbExamples, double* params, double gamma) {
        int i;
        double result = 0.0;
        
        for (i = 0; i < nbExamples; i++) {
            double distance = getDistance(params, inputs + (i * nbParameters), nbParameters);
            result += w[i] * exp(-gamma * abs(distance));
        }
        
        return result;
    }
}
