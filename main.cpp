#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/pick.h"
#include <igl/jet.h>
#include <Eigen/Sparse>

struct SimulationData
{
    int gridn; // The grid has gridn x gridn vertices    
    double gridh; // The width and height of each grid cell
    Eigen::MatrixXd V; // The grid vertex positions (used for rendering)
    Eigen::MatrixXi F; // The grid face indices (used for rendering)

    // Precomputed matrix decompositions
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > Lusolver; // Laplacian on the grid
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > Lvsolver; 
    
    // Simulation parameters
    double timestep;
    Eigen::Matrix<double, 9, 1> alphasu;
    Eigen::Matrix<double, 9, 1> betasu;
    Eigen::Matrix<double, 9, 1> alphasv;
    Eigen::Matrix<double, 9, 1> betasv;
    double Cu;
    double Cv;
    
    // State variables
    Eigen::VectorXd u; 
    Eigen::VectorXd v;
};

void buildLaplacian(SimulationData& simdata)
{
    int nedges = 2 * simdata.gridn * simdata.gridn;
    Eigen::SparseMatrix<double> D(nedges, simdata.gridn * simdata.gridn);
    std::vector<Eigen::Triplet<double> > Dcoeffs;
    for (int j = 0; j < simdata.gridn; j++)
    {
        for (int i = 0; i < simdata.gridn; i++)
        {
            int ip1 = (i+1) % simdata.gridn;
            Dcoeffs.push_back({ j * simdata.gridn + i, j * simdata.gridn + i, -1.0 });
            Dcoeffs.push_back({ j * simdata.gridn + i, j * simdata.gridn + ip1, 1.0 });
        }
    }
    for (int i = 0; i < simdata.gridn; i++)
    {
        for (int j = 0; j < simdata.gridn; j++)
        {
            int jp1 = (j+1) % simdata.gridn;
            Dcoeffs.push_back({ simdata.gridn * simdata.gridn + i * simdata.gridn + j, j * simdata.gridn + i, -1.0 });
            Dcoeffs.push_back({ simdata.gridn * simdata.gridn + i * simdata.gridn + j, jp1 * simdata.gridn + i, 1.0 });
        }
    }

    D.setFromTriplets(Dcoeffs.begin(), Dcoeffs.end());

    Eigen::SparseMatrix<double> L = (1.0 / simdata.gridh / simdata.gridh) * D.transpose() * D;
    
    Eigen::SparseMatrix<double> I(simdata.gridn*simdata.gridn, simdata.gridn*simdata.gridn);
    std::vector<Eigen::Triplet<double> > Icoeffs;
    for(int i=0; i<simdata.gridn*simdata.gridn; i++)
        Icoeffs.push_back({i,i,1.0});
    I.setFromTriplets(Icoeffs.begin(), Icoeffs.end());

    Eigen::SparseMatrix<double> Opu = I + simdata.timestep * simdata.Cu * L;
    Eigen::SparseMatrix<double> Opv = I + simdata.timestep * simdata.Cv * L;
    simdata.Lusolver.compute(Opu);
    simdata.Lvsolver.compute(Opv);
}

void makeGrid(int gridn, SimulationData& result)
{
    result.gridn = gridn;
    result.gridh = 2.0 / double(gridn - 1);
    result.V.resize(gridn * gridn, 3);
    for (int i = 0; i < gridn; i++)
    {
        for (int j = 0; j < gridn; j++)
        {
            double x = -1.0 + 2.0 * double(j) / double(gridn - 1);
            double y = -1.0 + 2.0 * double(i) / double(gridn - 1);
            result.V(i * gridn + j, 0) = x;
            result.V(i * gridn + j, 1) = y;
            result.V(i * gridn + j, 2) = 0;
        }
    }
    result.F.resize(2 * (gridn - 1) * (gridn - 1), 3);
    for (int i = 0; i < gridn - 1; i++)
    {
        for (int j = 0; j < gridn - 1; j++)
        {
            int idx = 2 * (i * (gridn - 1) + j);
            result.F(idx, 0) = i * gridn + j;
            result.F(idx, 1) = i * gridn + (j + 1);
            result.F(idx, 2) = (i + 1) * gridn + (j + 1);
            result.F(idx + 1, 0) = i * gridn + j;
            result.F(idx + 1, 1) = (i + 1) * gridn + (j + 1);
            result.F(idx + 1, 2) = (i + 1) * gridn + j;
        }
    }

    result.u.resize(gridn * gridn);
    result.u.setZero();
    
    result.v.resize(gridn * gridn);
    result.v.setZero();
    
    buildLaplacian(result);

}

void simulateOneStep(SimulationData& simdata)
{
    Eigen::VectorXd rhsu = simdata.u;
    Eigen::VectorXd rhsv = simdata.v;
    
    for(int i=0; i<simdata.gridn*simdata.gridn; i++)
    {
        double numu = 0.0;
        double denomu = 0.0;
        double numv = 0.0;
        double denomv = 0.0;
        for(int j=0; j<3; j++)
        {
            for(int k=0; k<3; k++)
            {
                numu += std::pow(simdata.u[i], j) * std::pow(simdata.v[i], k) * simdata.alphasu[3 * j + k];
                numv += std::pow(simdata.u[i], j) * std::pow(simdata.v[i], k) * simdata.alphasv[3 * j + k];
                denomu += std::pow(simdata.u[i], j) * std::pow(simdata.v[i], k) * simdata.betasu[3 * j + k];
                denomv += std::pow(simdata.u[i], j) * std::pow(simdata.v[i], k) * simdata.betasv[3 * j + k];
            }
        }
        if(std::fabs(denomu) > 1e-4)
            numu /= denomu;
        if(std::fabs(denomv) > 1e-4)
            numv /= denomv;
        rhsu[i] += simdata.timestep * numu;
        rhsv[i] += simdata.timestep * numv;
    }

    simdata.u = simdata.Lusolver.solve(rhsu);
    simdata.v = simdata.Lvsolver.solve(rhsv);    
}

int main(int argc, char *argv[])
{
    polyscope::init();

    SimulationData simdata;
    simdata.timestep = 0.001;
    simdata.Cu = 0.1;
    simdata.Cv = 0.1;
    simdata.alphasu[0] = 0.0;
    simdata.betasu[0] = 1.0;
    simdata.alphasv[0] = 0.0;
    simdata.betasv[0] = 1.0;
    for(int i=1; i<9; i++)
    {
        simdata.alphasu[i] = 0.0;
        simdata.betasu[i] = 0.0;
        simdata.alphasv[i] = 0.0;
        simdata.betasv[i] = 0.0;
    }

    makeGrid(100, simdata);

    // Set up rendering

    polyscope::view::style = polyscope::NavigateStyle::Planar;
    polyscope::view::projectionMode = polyscope::ProjectionMode::Orthographic;
    
    auto* pmesh = polyscope::registerSurfaceMesh("Mesh", simdata.V, simdata.F);
    auto *udata = pmesh->addVertexScalarQuantity("u", simdata.u);
    udata->setEnabled(true);
    pmesh->addVertexScalarQuantity("v", simdata.v);

    // GUI state

    polyscope::state::userCallback = [&]()->void
    {
        bool meshdirty = false;
        bool laplacianDirty = false;
        
        if(ImGui::Button("Randomize Morphogens"))
        {
            simdata.u.setRandom();
            simdata.v.setRandom();
        }
        
        if(ImGui::Button("Randomize Coefficients"))
        {
            simdata.alphasu.setRandom();
            simdata.alphasv.setRandom();
            simdata.betasu.setRandom();
            simdata.betasv.setRandom();            
        }

        int oldsize = simdata.gridn;
        if (ImGui::InputInt("Grid Size", &oldsize))
        {
            makeGrid(oldsize, simdata);
            meshdirty = true;
        }

        if (ImGui::InputDouble("Time step", &simdata.timestep))
        {
            laplacianDirty = true;
        }
        if(ImGui::InputDouble("Cu", &simdata.Cu))
        {
            laplacianDirty = true;
        }
        if(ImGui::InputDouble("Cv", &simdata.Cv))
        {
            laplacianDirty = true;
        }

        if (laplacianDirty)
        {
            buildLaplacian(simdata);
        }
        
        double vmin = -10.0;
        double vmax = 10.0;
        
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
            {
                std::stringstream ss;
                ss << "alpha_u[" << i << "][" << j << "]";
                ImGui::SliderScalar(ss.str().c_str(), ImGuiDataType_Double, &simdata.alphasu[3*i+j], &vmin, &vmax, "%.3f");
            }
        }
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
            {
                std::stringstream ss;
                ss << "beta_u[" << i << "][" << j << "]";
                ImGui::SliderScalar(ss.str().c_str(), ImGuiDataType_Double, &simdata.betasu[3*i+j], &vmin, &vmax, "%.3f");
            }
        }
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
            {
                std::stringstream ss;
                ss << "alpha_v[" << i << "][" << j << "]";
                ImGui::SliderScalar(ss.str().c_str(), ImGuiDataType_Double, &simdata.alphasv[3*i+j], &vmin, &vmax, "%.3f");
            }
        }
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
            {
                std::stringstream ss;
                ss << "beta_v[" << i << "][" << j << "]";
                ImGui::SliderScalar(ss.str().c_str(), ImGuiDataType_Double, &simdata.betasv[3*i+j], &vmin, &vmax, "%.3f");
            }
        }
        
        simulateOneStep(simdata);
        

        // Refresh all rendered geometry

        if (meshdirty)
        {
            pmesh = polyscope::registerSurfaceMesh("Mesh", simdata.V, simdata.F);
        }
        pmesh->addVertexScalarQuantity("u", simdata.u);
        pmesh->addVertexScalarQuantity("v", simdata.v);
    };
    
    polyscope::show();
}
