#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include <iostream>
#include "NeuralNetGraph.h"


TEST_CASE("Test Forward", "[NNGraph]"){
    std::vector<int> numNodesInLayer={2, 1};
    std::vector<double> x={.3, .5};
    auto nodes=nngraph::forward_prop_init(numNodesInLayer, x, [&](const auto& input){
        return 1.0/(1.0+exp(-input));
    }); 
    auto printLayer=[&](int layer, bool showW){
        for(int i=0; i<nodes[layer].size();++i){
            std::cout<<"oij: "<<nodes[layer][i]->oij<<std::endl;
            if(showW){
                for(int j=0;j<nodes[layer][i]->forwardW.size();++j){
                    std::cout<<"w: "<<nodes[layer][i]->forwardW[j].w<<std::endl;
                }
            }
            
        }
    };
    printLayer(0, true);
    printLayer(1, true);
    printLayer(2, false);


    nodes=nngraph::back_prop(std::move(nodes), .5,
    [&](const auto& x, const auto& y){
        return 2*(x-y);
    }
    );
    auto printLayerDW=[&](int layer, bool showW){
        for(int i=0; i<nodes[layer].size();++i){
            std::cout<<"oij: "<<nodes[layer][i]->oij<<std::endl;
            if(showW){
                for(int j=0;j<nodes[layer][i]->forwardW.size();++j){
                    std::cout<<"dw: "<<nodes[layer][i]->forwardW[j].dw<<std::endl;
                }
            }
        }
    };
    printLayerDW(0, true);
    printLayerDW(1, true);
    printLayerDW(2, true);
    nngraph::cleanup(nodes);
    //REQUIRE(newton::zeros(squareTestV, deriv, guess, .00001, 20)==Approx(sqrt(2.0)));
}  
  