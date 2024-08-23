//
// Created by polya on 8/23/24.
//
#include <valarray>

#include "potentialFunctionPrototype.hpp"
class V_inv_12_6:public potentialFunction
{
public:
    V_inv_12_6(const std::string &coefsStr):potentialFunction(){
        this->coefsInStr=coefsStr;
    }

    void json2Coefs(const std::string &coefsStr)override
    {
        std::stringstream iss;
        iss<<coefsStr;
        std::string temp;
        //read a1
        if (std::getline(iss, temp, ',')){
            this->a1=std::stod(temp);
        }

        //read b1

        if (std::getline(iss, temp, ',')){
            this->b1=std::stod(temp);
        }

        //read N

        if (std::getline(iss, temp, ',')){
            this->N=std::stoi(temp);
        }

    }

    void init() override
    {
        this->json2Coefs(coefsInStr);
        this->r1=std::pow(2.0*a1/b1,1.0/6.0);

        this->lm=(static_cast<double >(N)*r1)*3;

        std::cout << "a1=" << a1 << ", b1=" << b1  << std::endl;
        std::cout<<"r1="<<r1<<std::endl;
        std::cout<<"lm="<<lm<<std::endl;

    }
    double operator() (const double * xVec,const double * yVec, const int& j)override
    {//j is the index of the vertice to be updated
        double val=0;
        for(int i=0;i<4 and i!=j;i++)
        {
            double diff_xij=xVec[j]-xVec[i];
            double diff_yij=yVec[j]-yVec[i];
            double dist_ij=dist(diff_xij,diff_yij);
            val+=V1(dist_ij);
        }
        return val;
    }
    double potentialFull(const double * xVec,const double * yVec)override
    {
        double val=0;
        for(int i=0;i<4;i++)
        {
            for (int j=i+1;j<4;j++)
            {
                double diff_xij=xVec[j]-xVec[i];
                double diff_yij=yVec[j]-yVec[i];
                double dist_ij=dist(diff_xij,diff_yij);
                val+=V1(dist_ij);
            }//end j for
        }//end i for
        return val;
    }
    double dist(const double&x, const double &y)
    {
        return std::sqrt(std::pow(x,2.0)+std::pow(y,2.0));
    }
    double V1(const double &r){
        double  val=a1*std::pow(r,-12.0)-b1*std::pow(r,-6.0);
        return val;
    }

    double getLm() const override {
        return lm;
    }
public:
    double a1;
    // double a2;
    double b1;
    // double b2;
    std::string coefsInStr;
    double r1;//min position of V1
    // double r2;//min position of V2
    double lm;//range of distances
    int N;
};



std::shared_ptr<potentialFunction>  createPotentialFunction(const std::string& funcName, const std::string &coefsJsonStr) {
    if (funcName == "V_inv_12_6") {

        return std::make_shared<V_inv_12_6>(coefsJsonStr);
    }

    else {
        throw std::invalid_argument("Unknown potential function type");
    }
}