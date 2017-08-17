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
    double w; //stores the weight
    double dw; //stores the derivative of the weight
    double helperDW; //stores a helper function for the rest of the derivatives
    weights(){
        w=d(gen); //random weights
    }
};

/**this is a model for the nodes in a network.  */
/**The oij stores the value of 
g(w*theta) where w and theta are the 
outputs from prior nodes. dA is the
derivative of g at w*theta.
*/
struct Node {
    double oij;
    double dA;
    std::vector<weights> forwardW;
    void setWeights(int numNodes){
        forwardW=futilities::for_each_parallel(0, numNodes, [&](const auto& index){
            return weights();
        });
    }
    Node(int numNodes){
        setWeights(numNodes);
    }
    Node(int numNodes, double oij_, double dA_){
        oij=oij_;
        dA=dA_;
        setWeights(numNodes);
    }
    Node(double oij_, double dA_){
        oij=oij_;
        dA=dA_;
        setWeights(1);
    }

};

namespace nngraph{
    template<typename Fn>
    auto forward_prop_init(const std::vector<int>& numNodesInLayer, const std::vector<double>& x, const Fn& activation){
        std::vector<std::vector<Node*> > holdNodes;
        /**input layer*/
        holdNodes.emplace_back(futilities::for_each_parallel(0, (int)x.size(), [&](const auto& index){
            return new Node(numNodesInLayer[0], x[index], 0.0);
        }));
        int index=0;
        for_each(numNodesInLayer.begin(), numNodesInLayer.end(), [&](const auto& numNodes){
            index++;
            holdNodes.emplace_back(
                futilities::for_each_parallel(0, numNodes, [&](const auto& nodeIndex){
                    double inputIntoActivation=futilities::sum(holdNodes.back(), [&](const auto& nodePointerPrev, const auto& prevNodeIndex){
                        return nodePointerPrev->oij*nodePointerPrev->forwardW[nodeIndex].w;
                    });
                    auto dualActivation=activation(AutoDiff<double>(inputIntoActivation, 1.0));
                    if(index<numNodesInLayer.size()){
                        return new Node(numNodesInLayer[index], dualActivation.getStandard(), dualActivation.getDual());
                    }
                    else{
                        return new Node(dualActivation.getStandard(), dualActivation.getDual());
                    }
                })
            );
        });

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
                auto dualActivation=activation(AutoDiff<double>(inputIntoActivation, 1.0));
                node->oij=dualActivation.getStandard();
                node->dA=dualActivation.getDual();
                return node;
            });
        });
        return std::move(holdNodes);
    }


    template<typename Fn>
    auto back_prop(std::vector<std::vector<Node*> >&& holdNodes, const double& y, const Fn& costFunctionDeriv){

        //This is a "fake" derivative as there are no w's in the output layer in the graph model
        holdNodes.back()=futilities::for_each_parallel(std::move(holdNodes.back()), [&](auto& nodePointer, const auto& nodeIndex){
            nodePointer->forwardW=futilities::for_each_parallel(std::move(nodePointer->forwardW), [&](auto& w, const auto& wIndex){
                w.helperDW=costFunctionDeriv(nodePointer->oij, y)*nodePointer->dA;
                return w;
            });
            return nodePointer;
        });
        int index=holdNodes.size();
        for_each(holdNodes.rbegin()+1, holdNodes.rend(), [&](auto& layers){
            index--;
            layers=futilities::for_each_parallel(std::move(layers), [&](auto& nodePointer, const auto& nodeIndex){
                auto refToForwardNode=holdNodes[index];//reference to the layer one closer to output
                nodePointer->forwardW=futilities::for_each_parallel(std::move(nodePointer->forwardW), [&](auto& w, const auto& wIndex){
                    //the dW is equal to the previous helper (a chain rule property), multiplied by the current value.  note that the current value is equivalent to the "theta*w" , the derivative of which is theta.
                    w.dw=futilities::sum(refToForwardNode[wIndex]->forwardW, [&](const auto& ws, const auto& wsIndex){
                        return ws.helperDW;
                    });
                    //continues propagating the chain rule. Uses the current activation function derivative multipled by the current w...since the derivatives not yet computed need the current w 
                    w.helperDW=nodePointer->dA*w.dw*w.w;

                    w.dw*=nodePointer->oij;
                    return w;
                });
                return nodePointer;
            });
        });
        return std::move(holdNodes);
    }
    void cleanup(std::vector<std::vector<Node*> >& holdNodes){
        for(auto&& nodeV:holdNodes){
            for(auto&& node:nodeV){
                delete node;
            }
        }
        holdNodes.clear();
    }

}


#endif