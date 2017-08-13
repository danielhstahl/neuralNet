#ifndef NEURALNET_H
#define NEURALNET_H
#include "FunctionalUtilities.h"
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace neuralnet{
    /**Here "theta" is an m*p matrix where m is the number of features in step j and p is the number of features in step j+1.  "a" is a n by m matrix where n is the number of samples and m the number of features*/
    template<typename Number>
    auto computeThetaByA(const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> theta , const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> a){
        return a*theta;
    }
    /**Here "x" is a n by m matrix where n is the number of samples and m the number of features */
    template<typename Number, FN>
    auto forward_prop(const FN& sigmoid, Eigen::Matrix<Number, Eigen::Dynamic,  Eigen::Dynamic>& x, const Eigen::Matrix<Number,Eigen::Dynamic, Eigen::Dynamic>& theta, const Eigen::Matrix<Number,Eigen::Dynamic, Eigen::Dynamic>&... thetas){
        //x.conservativeResize(x.rows(), x.cols()+1);
        auto vec1=Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>::Ones(x.rows(), 1);
        x<<vec1;
        return forwardprop(computeThetaByA(theta, x).unaryExpr(&sigmoid), thetas); 
    }
    
    template<typename Number, FN>
    auto backward_prop(const FN& sigmoid, Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>& x, Eigen::Matrix<Number, Eigen::Dynamic, 1>& y, const Eigen::Matrix<Number,Eigen::Dynamic, Eigen::Dynamic>& theta, const Eigen::Matrix<Number,Eigen::Dynamic, Eigen::Dynamic>&... thetas){
        
    }
    /**Features is the number of features in X. Layers is the number of hidden layers in the nn.  Units is the number of "features" in each hidden layer.  Labels is the total possible outcomes for the NN*/
    template<int Features=0, int Layers=0, int Units=0, int Labels=0, typename T>
    struct NN {
        typedef Eigen::Matrix<T, Eigen::Dynamic, Features> X; //explanatory data
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Y; //result data
        typedef Eigen::Matrix<T, Eigen::Dynamic, Labels> categorizedY; //result data
        typedef Eigen::Matrix<T, Units, Features+1> initialTheta; //add a "bias" term
        typedef Eigen::Matrix<T, Units, Units+1> standardTheta;
        typedef Eigen::Matrix<T, Labels, Units+1> endTheta;
        typedef Eigen::Matrix<T, Units*(Features+1)+Units*(Units+1)*Layers+Labels*(Units+1), 1> allTheta;

        typedef Eigen::Matrix<T, Eigen::Dynamic, Units+1> standardA;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Labels> endA;


        /**Here "theta" is an m*p matrix where m is the number of features in step j and p is the number of features in step j+1.  "a" is a n by m matrix where n is the number of samples and m the number of features*/
        template<typename Mat1, typename Mat2>
        auto computeThetaByA(const Mat1& theta , const Mat2& a){
            return a*theta.transpose();
        }
        /**Here "x" is a n by m matrix where n is the number of samples and m the number of features */
        template<typename Number, FN>
        auto forward_prop(const FN& sigmoid,  const standardA& a){
            return a; 
        }
        /**Here "x" is a n by m matrix where n is the number of samples and m the number of features */
        template<typename Number, FN>
        auto forward_prop(const FN& sigmoid,  const standardA& a, const standardTheta& theta, const standardTheta&... thetas){
            //x.conservativeResize(x.rows(), x.cols()+1);
            auto vec1=Eigen::Matrix<Number, Eigen::Dynamic, 1>::Ones(x.rows());
            x<<vec1;
            return forward_prop(sigmoid, computeThetaByA(theta, x).unaryExpr(&sigmoid), thetas); 
        }

        NN(){}//do nothing on construct
        execute(X&& x, const Y& y, const allTheta& theta){
            x<<Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(y.rows());//append column of ones
            /*convert y to category vectors*/
            categorizedY y_h=futilities::for_each_parallel_generic(
                [](auto&& matrix){return 0;}, 
                [](auto&& matrix){return matrix.rows();}, 
                categorizedY::Zero(y.rows(), Labels), 
                [&](const auto& index, auto&& matrix){
                    matrix(i, y(i))=1;
                }
            );
            Map<allTheta> theta1(theta.data(),  Units, Features+1);
            auto a2=computeThetaByA(theta, x).unaryExpr(&sigmoid);
            for(int i=0; i<Layers; ++i){
                computeThetaByA(Map<allTheta> theta1(theta.data(),  Units, Features+1), a2).unaryExpr(&sigmoid);
                a*theta.transpose();
            }


            forward_prop

        }
            

    };

};
#endif