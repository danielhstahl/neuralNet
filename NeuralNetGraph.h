#ifndef NNGRAPH_H
#define NNGRAPH_H

#include "FunctionalUtilities.h"
#include "AutoDiff.h"
#include <random>

std::random_device rd;
std::mt19937 gen(rd());

// values near the mean are the most likely
// standard deviation affects the dispersion of generated values from the mean
std::normal_distribution<> d(0,2);



//totalToThisPoint+totalInCurrentIndex
int numFromEnd(int totalNodes, int totalToThisPoint, int totalInCurrentIndex){
    return totalNodes-totalToThisPoint-totalInCurrentIndex;
}

struct weights{
    double w;
    double dw;
    weights(){
        w=d(gen);
    }
};

struct Node {
    double oij;
    std::vector<weights> forwardW;
    void setWeights(int numNodes){
        forwardW=futilities::for_each_parallel(0, numNodes, [&](const auto& index){
            return weights();
        });
    }
    Node(int numNodes){
        setWeights(numNodes);
    }
    Node(int numNodes, double oij_){
        oij=oij_;
        setWeights(numNodes);
    }
    Node(double oij_){
        oij=oij_;
        setWeights(1);
    }

};

namespace nngraph{
    template<typename Fn>
    auto forward_prop_init(const std::vector<int>& numNodesInLayer, const std::vector<double>& x, const Fn& activation){
        std::vector<std::vector<Node*> > holdNodes;
        /**input layer*/
        holdNodes.emplace_back(futilities::for_each_parallel(0, (int)x.size(), [&](const auto& index){
            return new Node(numNodesInLayer[0], x[index]);
        }));
        int index=0;
        for_each(numNodesInLayer.begin(), numNodesInLayer.end()-1, [&](const auto& numNodes){
            index++;
            holdNodes.emplace_back(
                futilities::for_each_parallel(0, numNodes, [&](const auto& nodeIndex){
                    double inputIntoActivation=futilities::sum(holdNodes.back(), [&](const auto& nodePointerPrev, const auto& prevNodeIndex){
                        return nodePointerPrev->oij*nodePointerPrev->forwardW[nodeIndex].w;
                    });
                    return new Node(numNodesInLayer[index], activation(inputIntoActivation));
                })
            );
        });
        holdNodes.emplace_back(
            futilities::for_each_parallel(0, numNodesInLayer.back(), [&](const auto& nodeIndex){
                double inputIntoActivation=futilities::sum(holdNodes.back(), [&](const auto& nodePointerPrev, const auto& prevNodeIndex){
                    return nodePointerPrev->oij*nodePointerPrev->forwardW[nodeIndex].w;
                });
                return new Node(activation(inputIntoActivation));
            })
        );
        return holdNodes;
    }
    template<typename Fn>
    auto forward_prop(std::vector<std::vector<Node*> > holdNodes, const std::vector<double>& x, const Fn& activation){
        /**input layer*/
        holdNodes[0]=futilities::for_each_parallel(std::move(holdNodes[0]), [&](auto&& node, const auto& index){
            node->oij=x[index];
        });

        /**forward propogate*/
        return futilities::for_each_subset(std::move(holdNodes), 1, 0, [&](auto&& layer, const auto& layerIndex){
            return futilities::for_each_parallel(std::move(layer), [&](auto&& node, const auto& nodeIndex){
                double inputIntoActivation=futilities::sum(holdNodes[layerIndex-1], [&](const auto& nodePointerPrev, const auto& prevNodeIndex){
                    return nodePointerPrev->oij*nodePointerPrev->forwardW[nodeIndex].w;
                });
                node->oij=activation(inputIntoActivation);
                return node;
            });
        });
        return std::move(holdNodes);
    }


    /**This isn't done yet*/
    template<typename Fn, typename Act>
    auto back_prop(std::vector<std::vector<Node*> >&& holdNodes, const double& y, const Act& activation, const Fn& costFunctionDeriv){

        //This is a "fake" derivative as there are no w's in the output layer in the graph model
        holdNodes.back()=futilities::for_each_parallel(std::move(holdNodes.back()), [&](auto& nodePointer, const auto& nodeIndex){
            nodePointer->forwardW=futilities::for_each_parallel(std::move(nodePointer->forwardW), [&](auto& w, const auto& wIndex){
                w.dw=costFunctionDeriv(nodePointer->oij, y)*activation(AutoDiff<double>(nodePointer->oij, 1.0)).getDual();
                return w;
            });
            return nodePointer;
        });
        int index=holdNodes.size();
        for_each(holdNodes.rbegin()+1, holdNodes.rend(), [&](auto& layers){
            index--;
            //std::cout<<layers.size()<<std::endl;
            layers=futilities::for_each_parallel(std::move(layers), [&](auto& nodePointer, const auto& nodeIndex){
                double mutualDerivative=activation(AutoDiff<double>(nodePointer->oij, 1.0)).getDual()*nodePointer->oij;
                auto refToForwardNode=holdNodes[index];//reference to the layer one closer to output
                nodePointer->forwardW=futilities::for_each_parallel(std::move(nodePointer->forwardW), [&](auto& w, const auto& wIndex){
                    w.dw=w.w*mutualDerivative*refToForwardNode[wIndex]->forwardW[wIndex].dw/refToForwardNode[wIndex]->oij;
                    return w;
                });
                return nodePointer;
            });
        });
        
        return std::move(holdNodes);
    }

}


#endif