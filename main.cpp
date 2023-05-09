#include <iostream>
#include <random>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>

//=============================== PARAMETER DEFINITIONS ==========================//

int L = 1280; // If Q=7 this must be an even number!
int rows = L;
int cols = 100;
const int Q = 7; // Which lattice type to use (7 or 9)
const int sweeps = 50e3; // Number of time-steps to perform
const int save_sweep = 100; // How often to save data
const double tol = 1e-10;
const double init_rad = 25.; // Radius of cell layer (if using circular geometry)
const double pi = M_PI;
const double density = 0.1; // Average initial fluid density
const double rho_star = 0.15; // Critical density
double rho_c = 0.05; // Threshold density (Minimum density allowed at given site)
const double sigma = 0.01; // Noise strength
const double tau = 1.; // Relaxation time
double g = 0.001; // Strength of substrate friction force
double P_division = 0.001; // Division rate
const int BB_INTERFACE = 1; // Switch for implementing interface bounce-back
const int PERIODIC = 0; // Switch for implementing periodic boundary conditions
const int DOMAIN_ADAPTION = 1; // Switch for implementing variable domain size

//===============================================================================//

using namespace std;
using namespace arma;

void simulation();
void initialisation_circle(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double> > > &f, vector<vector<double> >  &e_k, vector<double> &w, mt19937 &mt_rand);
void initialisation_planar(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double> > > &f, vector<vector<double> >  &e_k, vector<double> &w, mt19937 &mt_rand);
void stream(vector<vector<double> > &rho, vector<vector<vector<double> > > &f, vector<vector<vector<double> > > &e_k_stream);
void collision(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double > > > &f, vector<vector<vector<double > > > &feq);
void equilibrium(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double > > > &feq, vector<vector<double> > &e_k, vector<double> &w);
void compute_rho_and_vel(vector<vector<vector<double> > > &f, vector<vector<double> > &rho, vector<vector<vector<double> > > &v, vector<vector<double> > &e_k);
void add_noise(vector<vector<double> > &rho, vector<vector<vector<double> > > &f, mt19937 &mt_rand);
void shan_chen(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double > > > &v_eq, vector<vector<vector<double> > > &f, vector<vector<vector<double> > > &e_k_stream, vector<vector<double> > &e_k, vector<double> &w);
void cell_division(vector<vector<vector<double> > > &f, vector<vector<double> > &rho, mt19937 &mt_rand);
void variable_saver(vector<vector<double> > &rho, vector<vector<vector<double> > > &v, int &t);
void boundary_saver(vector<vector<int> > &b, int &t);
void compute_boundary_width(vector<vector<double> > &rho, vector<double> &boundary_width, int &t);
void boundary_width_saver(vector<vector<double> > &rho, vector<double> &boundary_width);
void domain_adapter(vector<vector<double> > &rho, vector<vector<vector<double> > > &f, vector<vector<vector<double> > > &feq, vector<vector<vector<double> > > &v, vector<vector<vector<double> > > &v_SC, vector<double> &boundary_width, int &cols, int &t);
void cell_division_variable(vector<vector<vector<double> > > &f, vector<vector<double> > &rho, mt19937 &mt_rand, int &t, vector<double> &prolif_differential);
void cell_division_density_dependent(vector<vector<vector<double> > > &f, vector<vector<double> > &rho, mt19937 &mt_rand, int &t, vector<vector<double> > &prolif_differential);
void boundary_tracker(vector<vector<double> > &rho, vector<vector<int> > &b, vector<double> &boundary_width, int &t);

int main()
{
    //========== RUN SIMULATION ==========//
    simulation();
    return 0;
}

void simulation()
{
    // Initialise data structures
    vector<vector<vector<double> > > f(rows,vector<vector<double> >(cols,vector<double>(Q,0)));
    vector<vector<vector<double> > > feq(rows,vector<vector<double> >(cols,vector<double>(Q,0)));
    vector<vector<double> > rho(rows,vector<double>(cols,0));
    vector<vector<vector<double> > > v(rows,vector<vector<double> >(cols,vector<double>(2,0)));
    vector<vector<vector<double> > > v_star(rows,vector<vector<double> >(cols,vector<double>(2,0)));
    vector<vector<vector<double> > > v_SC(rows,vector<vector<double> >(cols,vector<double>(2,0)));
    vector<vector<double> > e_k(2,vector<double>(Q,0));
    vector<vector<vector<double> > > e_k_stream(2,vector<vector<double> >(2,vector<double>(Q,0)));
    vector<double> w(Q,0);
    vector<double> boundary_width(sweeps,0);
    vector<vector<double> > prolif_differential(sweeps,vector<double>(3,0));
    vector<vector<int> > b(2*L,vector<int>(2,0));
    // Initialise mersenne twister generator
    random_device device;
    mt19937 mt_rand(device());

    // Initialise weights and lattice directions
    if(Q==7)
    {
        e_k = {{0.,0.},{1.,0.},{0.5,0.5*sqrt(3)},{-0.5,0.5*sqrt(3)},{-1.,0.},{-0.5,-0.5*sqrt(3)},{0.5,-0.5*sqrt(3)}};

        vector<vector<double> > e_k_se{{0.,0.},{1.,0.},{0.,1.},{-1.,1.},{-1.,0.},{-1.,-1.},{0.,-1.}};
        vector<vector<double> > e_k_so{{0.,0.},{1.,0.},{1.,1.},{0.,1.},{-1.,0.},{0.,-1.},{1.,-1.}};
        e_k_stream = {e_k_se,e_k_so};

        w = {1./2.,1./12.,1./12.,1./12.,1./12.,1./12.,1./12.};
    }
    else if(Q==9)
    {
        e_k = {{0.,0.},{1.,0.},{1.,1.},{0.,1.},{-1.,1.},{-1.,0.},{-1.,-1.},{0.,-1.},{1.,-1.}};

        e_k_stream = {e_k,e_k};

        w = {4./9.,1./9.,1./36.,1./9.,1./36.,1./9.,1./36.,1./9.,1./36.};
    }

    // Initialise distribution lattice and macroscopic fields and distribution f
    initialisation_planar(rho,v,f,e_k,w,mt_rand);

    int t = 0;
    // Main time loop
    while(t < sweeps)
    {
        // Detect tissue boundary using left-moving walk
        boundary_tracker(rho,b,boundary_width,t);
        // Save variables
        if((t%save_sweep == 0) && (t >= 0.5*sweeps))
        {
            boundary_saver(b,t);
            variable_saver(rho,v,t);
        }
        // Stream
        stream(rho,f,e_k_stream);

        // Inject mass into the tissue via cell division
        cell_division(f,rho,mt_rand,t,prolif_differential);

        // Calculate macroscopic variables
        compute_rho_and_vel(f,rho,v,e_k);

        // Calculate equilibrium distribution
        equilibrium(rho,v,feq,e_k,w);

        // Collision
        collision(rho,v,f,feq);

        // Add noise
        add_noise(rho,f,mt_rand);

        // Cut off part of the tissue bulk to keep domain size down
        if(DOMAIN_ADAPTION)
        {
            domain_adapter(rho,f,feq,v,v_SC,boundary_width,cols,t);
        }

        // Increment time by one
        t++;
    }
    //Save boundary width through time
    boundary_width_saver(rho, boundary_width);
}

void initialisation_planar(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double> > > &f, vector<vector<double> > &e_k, vector<double> &w, mt19937 &mt_rand)
{
    uniform_real_distribution<double> dens_distrib(0.9*density,1.1*density);
    uniform_real_distribution<double> vel_distrib(0.,2*pi);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if(j<0.5*cols)
            {
                double c = 1.0 * j;
                rho[i][j] = dens_distrib(mt_rand);
                int angle = vel_distrib(mt_rand);
                v[i][j][0] = sigma*cos(angle); v[i][j][1] = sigma*sin(angle);
            }else
            {
                rho[i][j] = 0.;
                v[i][j][0] = 0.; v[i][j][1] = 0.;
            }
        }
    }
    equilibrium(rho,v,f,e_k,w);
}

void initialisation_circle(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double> > > &f, vector<vector<double> > &e_k, vector<double> &w, mt19937 &mt_rand)
{
    uniform_real_distribution<double> dens_distrib(0.9*density,1.1*density);
    uniform_real_distribution<double> vel_distrib(0.,2*pi);
    double centre = (L/2.-1);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double rad_dist = sqrt((i-centre)*(i-centre) + (j-centre)*(j-centre));
            if(rad_dist<init_rad)
            {
                rho[i][j] = dens_distrib(mt_rand);
                int angle = vel_distrib(mt_rand);
                v[i][j][0] = 0.; v[i][j][1] = 0.;
            }else
            {
                rho[i][j] = 0.;
                v[i][j][0] = 0.; v[i][j][1] = 0.;
            }
        }
    }

    //active_mips(rho,v,e_k);
    equilibrium(rho,v,f,e_k,w);
}

void stream(vector<vector<double> > &rho, vector<vector<vector<double> > > &f, vector<vector<vector<double> > > &e_k_stream)
{
    int inew,jnew;
    vector<vector<vector<double> > > f_buff(rows,vector<vector<double> >(cols,vector<double>(Q)));

    vector<int> noslip(Q);
    if(Q == 7){noslip = {0,4,5,6,1,2,3};}
    else if(Q == 9){noslip = {0,5,6,7,8,1,2,3,4};}

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < Q; k++)
            {
                int r = i % 2;
                inew = i + e_k_stream[r][k][1];
                if (inew < 0){inew = rows - 1;}
                else if (inew >= rows){inew = 0;}
                jnew = j + e_k_stream[r][k][0];
                if (jnew < 0){jnew = cols - 1;}
                else if (jnew >= cols){jnew = 0;}
                switch(PERIODIC) // If non-periodic boundary conditions implement bounceback for edge sites
                {
                    case 0 :
                        if ((j == 0) && (e_k_stream[r][k][0]==-1)) // Don't stream through wall nodes on left-wall, implement bounceback
                        {
                            f_buff[i][j][noslip[k]] = f[i][j][k];
                            continue;
                        }
                        else if ((j == cols - 1) && (e_k_stream[r][k][0]==1)) // Don't stream through wall nodes on right-wall, implement bounceback
                        {
                            f_buff[i][j][noslip[k]] = f[i][j][k];
                            continue;
                        }
                        else
                        {
                            f_buff[inew][jnew][k] = f[i][j][k];
                        }
                        break;
                    case 1:
                        f_buff[inew][jnew][k] = f[i][j][k];
                        break;
                }
            }
        }
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < Q; k++)
            {
                f[i][j][k] = f_buff[i][j][k];
            }
        }
    }
}

void cell_division(vector<vector<vector<double> > > &f, vector<vector<double> > &rho, mt19937 &mt_rand)
{
    vector<int> noslip(Q);
    if(Q == 7){noslip = {0,4,5,6,1,2,3};}
    else if(Q == 9){noslip = {0,5,6,7,8,1,2,3,4};}

    double sum, mass, delta_mass, no_of_sites; int i, j, k;
    vector<vector<double> > dens(rows,vector<double>(cols,0));
    vector<double> b(rows,0);

    int edge_r = 0;
    int edge_l = 500;
    for (i = 0; i < rows; i++)
    {
        int c = 0;
        for (j = 0; j < cols; j++)
        {
            sum = 0.;
            for (k = 0; k < Q; k++)
            {
                sum = sum + f[i][j][k];
            }
            dens[i][j] = sum;

            if(dens[i][j] < tol && c == 0)
            {
                if(j > edge_r)
                {
                    edge_r = j;
                }
                c = 1;

                b[i] = j;
            }

            mass = mass + dens[i][j];
        }
    }
    delta_mass = P_division*mass;
    no_of_sites = ceil(delta_mass/0.01);
    no_of_sites = 1.0 * no_of_sites;

    no_of_sites = min(no_of_sites, P_division*rho_star*L*50/0.01);

    uniform_int_distribution<int> row_distrib(0,rows-1);
    uniform_int_distribution<int> col_distrib(0,edge_r);
    uniform_int_distribution<int> direction_distrib(1,0.5*(Q-1));
    for(int l = 0; l < no_of_sites; l++)
    {
        i = row_distrib(mt_rand);
        j = col_distrib(mt_rand);
        while(dens[i][j]<rho_c)
        {
            i = row_distrib(mt_rand);
            j = col_distrib(mt_rand);
        }

        if(Q == 9)
        {
            for (int k = 0; k < Q; k++)
            {
                f[i][j][k]=f[i][j][k] + delta_mass/(no_of_sites*Q);
            }
        }
        else if(Q == 7)
        {
            int dir = direction_distrib(mt_rand);
            f[i][j][dir]=f[i][j][dir] + delta_mass/(no_of_sites*2.);
            f[i][j][dir+0.5*(Q-1)]=f[i][j][dir+0.5*(Q-1)] + delta_mass/(no_of_sites*2.);
        }
    }
}

void compute_rho_and_vel(vector<vector<vector<double> > > &f, vector<vector<double> > &rho, vector<vector<vector<double> > > &v, vector<vector<double> > &e_k)
{
    double sum; vector<double> buffer(2);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            sum = 0.; buffer[0] = 0.; buffer[1] = 0.;
            for (int k = 0; k < Q; k++)
            {
                sum = sum + f[i][j][k];
                buffer[0] = buffer[0] + e_k[k][0] * f[i][j][k]; buffer[1] = buffer[1] + e_k[k][1] * f[i][j][k];
            }
            rho[i][j] = sum;
            if (rho[i][j] > tol)
            {
                v[i][j][0] = buffer[0]/rho[i][j]; v[i][j][1] = buffer[1]/rho[i][j];
            }
            else
            {
                v[i][j][0] = 0; v[i][j][1] = 0;
                break;
            }
        }
    }
}

void equilibrium(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double > > > &feq, vector<vector<double> > &e_k, vector<double> &w)
{
    double vdotek,vsqrd;
    double v_rms=0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            v[i][j][0] *= (1. - g); v[i][j][1] *= (1. - g);
            vsqrd = v[i][j][0]*v[i][j][0] + v[i][j][1]*v[i][j][1];
            v_rms += vsqrd;
            for (int k = 0; k < Q; k++)
            {
                vdotek = (v[i][j][0] * e_k[k][0]) + (v[i][j][1] * e_k[k][1]);
                if (Q==7){feq[i][j][k] = rho[i][j] * w[k] * (1. + 4.*vdotek + 8.*vdotek*vdotek - 2.*vsqrd);}
                else if (Q==9){feq[i][j][k] = rho[i][j] * w[k] * (1. + 3.*vdotek + (9./2.)*vdotek*vdotek - (3./2.)*vsqrd);}
            }
        }
    }
}

void collision(vector<vector<double> > &rho, vector<vector<vector<double > > > &v, vector<vector<vector<double > > > &f, vector<vector<vector<double > > > &feq)
{
    vector<int> noslip(Q);
    if(Q == 7){noslip = {0,4,5,6,1,2,3};}
    else if(Q == 9){noslip = {0,5,6,7,8,1,2,3,4};}

    vector<double> buffer(Q);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            switch(BB_INTERFACE) // If hard interface is switched on implement bounceback for sites less than threshold density
            {
                case 0 :
                    for (int k = 0; k < Q; k++)
                    {
                        f[i][j][k] = f[i][j][k] - ((1./tau) * (f[i][j][k] - feq[i][j][k]));
                    }
                    break;
                case 1 :
                    if (rho[i][j] < rho_c)
                    {
                        for (int k = 0; k < Q; k++)
                        {
                            buffer[k] = f[i][j][noslip[k]];
                        }
                        for (int k = 0; k < Q; k++)
                        {
                            f[i][j][k] = buffer[k];
                        }
                    }
                    else
                    {
                        for (int k = 0; k < Q; k++)
                        {
                            f[i][j][k] = f[i][j][k] - ((1./tau) * (f[i][j][k] - feq[i][j][k]));
                        }
                    }
                    break;
            }
        }
    }
}

void add_noise(vector<vector<double> > &rho, vector<vector<vector<double> > > &f, mt19937 &mt_rand)
{
    vector<int> noslip(Q);
    if(Q == 7){noslip = {0,4,5,6,1,2,3};}
    else if(Q == 9){noslip = {0,5,6,7,8,1,2,3,4};}

    double dens,sum,correction;
    uniform_real_distribution<double> real(-1,1);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if(PERIODIC==0 && (j==0 || j==cols-1)) // Do not apply noise at the walls - *THIS IS WHEN USING HARD BOUNDARY CONDITIONS*
            {
                continue;
            }
            dens = 0;
            for (int k = 0; k < Q; k++)
            {
                dens = dens + f[i][j][k];
            }

            if (dens > rho_c) // Only add noise to occupied sites
            {
                for (int k = 0; k < Q; k++)
                {
                    f[i][j][k] = f[i][j][k] + sigma * real(mt_rand);
                }
            }

            for (int k = 0; k < Q; k++)
            {
                if (f[i][j][k] < 0.)
                {
                    if (k == 0){f[i][j][k] = 0.;}
                    else
                    {
                        correction = -f[i][j][k];
                        f[i][j][noslip[k]] = f[i][j][noslip[k]] + correction;
                        f[i][j][k] = 0.;
                    }
                }
            }
            // Re-scale distribution to account for any change in density
            sum = 0;
            for (int k = 0; k < Q; k++)
            {
                sum = sum + f[i][j][k];
            }
            if (sum>tol)
            {
                for (int k = 0; k < Q; k++)
                {
                    f[i][j][k] = (dens/sum) * f[i][j][k];
                }
            }
        }
    }
}

void domain_adapter(vector<vector<double> > &rho, vector<vector<vector<double> > > &f, vector<vector<vector<double> > > &feq, vector<vector<vector<double> > > &v, vector<vector<vector<double> > > &v_SC, vector<double> &boundary_width, int &cols, int &t)
{
    w = boundary_width[t];

    if(cols < max(min_cols, floor(20*w)))
    {
        old_cols = cols;
        cols = max(min_cols, floor(20*w));

        for (int i = 0; i < rows; i++)
        {
            f[i].resize(cols,vector<double>(Q));
            feq[i].resize(cols,vector<double>(Q));
            rho[i].resize(cols,0);
            v[i].resize(cols,vector<double>(2,0));
            v_SC[i].resize(cols,vector<double>(2,0));
            for (int j = old_cols; j < cols-1; j++)
            {
                for (int k = 0; k < Q; k++)
                {
                    f[i][j][k] = 0.;
                }
            }
        }
    }

    vector<vector<vector<double> > > f_buff(rows,vector<vector<double> >(cols,vector<double>(Q)));

    if (h_ave > max(0.8*min_cols, floor(16*w)))
    {

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols-1; j++)
            {
                jnew = j + max(min_cols/5, floor(4*w));
                for (int k = 0; k < Q; k++)
                {
                    if (jnew < cols-1)
                    {
                        f_buff[i][j][k] = f[i][jnew][k];
                    }
                    else
                    {
                        f_buff[i][j][k] = 0.;
                    }
                }
            }
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sum = 0;
                for (int k = 0; k < Q; k++)
                {
                    f[i][j][k] = f_buff[i][j][k];
                    sum = sum + f[i][j][k];
                }
                rho[i][j] = sum;
            }
        }
    }

}

void boundary_width_saver(vector<vector<double> > &rho, vector<double> &boundary_width)
{
    string boundaryw_name = "./boundary_width2.txt";
    const char* final_boundaryw_name = boundaryw_name.c_str();
    FILE *fp=fopen(final_boundaryw_name,"w");
    for (int i = 0; i < sweeps; i++)
    {
        fprintf(fp,"%g\n",boundary_width[i]);
    }
    fclose(fp);
}

void boundary_saver(vector<vector<int> > &b, int &t)
{
    ostringstream ss; ss << setw(7) << setfill('0') << t;
    string number = ss.str();
    string boundary_name = "./BoundaryFiles/boundary" + number + ".txt";
    const char* final_boundary_name = boundary_name.c_str();
    FILE *fp=fopen(final_boundary_name,"w");
    for (int i = 0; i < 2*L; i++)
    {
        if(b[i][0]==2*L && b[i][1]==2*L)
        {
            break;
        }

        double r = 0.5 * sqrt(3) * b[i][0];
        double c = 1.0 * b[i][1];
        if(i % 2 == 0)
        {
            fprintf(fp,"%g %g\n",r,c);
        }else
        {
            fprintf(fp,"%g %g\n",r,c + 0.5);
        }
    }
    fclose(fp);
}

void variable_saver(vector<vector<double> > &rho, vector<vector<vector<double> > > &v, int &t)
{
    ostringstream ss; ss << setw(7) << setfill('0') << t;
    string number = ss.str();
    string density_name = "./DensityFiles/rho" + number + ".txt";
    const char* final_density_name = density_name.c_str();
    FILE *fp=fopen(final_density_name,"w");
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if(Q == 7)
            {
                double r = 0.5 * sqrt(3) * i;
                double c = 1.0 * j;

                if(rho[i][j]>tol)
                {
                    if(i % 2 == 0)
                    {
                        fprintf(fp,"%g %g %g %g\n",r,c,rho[i][j],atan2(v[i][j][1],v[i][j][0]));
                    }else
                    {
                        fprintf(fp,"%g %g %g %g\n",r,c+0.5,rho[i][j],atan2(v[i][j][1],v[i][j][0]));
                    }
                }
                else
                {
                    if(i % 2 == 0)
                    {
                        fprintf(fp,"%g %g %g %g\n",r,c,rho[i][j],0);
                    }else
                    {
                        fprintf(fp,"%g %g %g %g\n",r,c+0.5,rho[i][j],0);
                    }
                }

            }
            else if(Q == 9)
            {
                if(rho[i][j]>tol)
                {
                    fprintf(fp,"%d %d %g %g\n",i,j,rho[i][j],atan2(v[i][j][1],v[i][j][0]));
                }
                else
                {
                    fprintf(fp,"%d %d %g %g\n",i,j,rho[i][j],0);
                }
            }
        }
    }
    fclose(fp);

}

void boundary_tracker(vector<vector<double> > &rho, vector<vector<int> > &b, vector<double> &boundary_width, int &t)
{
    vector<vector<double> > e_k_se{{-1.,0.},{-1.,1.},{0.,1.},{1.,0.},{0.,-1.},{-1.,-1.}};
    vector<vector<double> > e_k_so{{-1.,0.},{0.,1.},{1.,1.},{1.,0.},{1.,-1.},{0.,-1.}};
    vector<vector<vector<double> > > e_k_stream = {e_k_se,e_k_so};

    int inew,jnew,r,c,n=0;
    for (int j = 0; j < cols; j++)
    {
        if(rho[0][j] > rho_c){continue;}
        else
        {
            r = 0;
            c = j;
            b[n][0] = r;
            b[n][1] = c;
            n++;
            break;
        }
    }

    int k_prev;
    for (int k = 0; k < 6; k++)
    {
        int x = r % 2;
        inew = r + e_k_stream[x][k][1];
        if (inew < 0){inew = rows - 1;}
        else if (inew >= rows){inew = 0;}
        jnew = c + e_k_stream[x][k][0];
        if (jnew < 0){jnew = cols - 1;}
        else if (jnew >= cols){jnew = 0;}

        if (rho[inew][jnew]<rho_c)
        {
            r = inew;
            c = jnew;
            b[n][0] = r;
            b[n][1] = c;
            k_prev = k;
            n++;
            break;
        }
    }

    while((r != 0) || (c != b[0][1]))
    {
        int k_start = k_prev - 2;
        if(k_start < 0){k_start = 6 + k_start;}

        for (int k = 0; k < 6; k++)
        {
            int k_current = k_start + k;
            if (k_current > 5){k_current = k_current - 6;}

            int x = r % 2;
            inew = r + e_k_stream[x][k_current][1];
            if (inew < 0){inew = rows - 1;}
            else if (inew >= rows){inew = 0;}
            jnew = c + e_k_stream[x][k_current][0];
            if (jnew < 0){jnew = cols - 1;}
            else if (jnew >= cols){jnew = 0;}

            if (rho[inew][jnew]<rho_c)
            {
                k_prev = k_current;

                r = inew;
                c = jnew;

                int new_point = 1;
                for(int a=0;a<n;a++)
                {
                    if(b[a][0] == inew && b[a][1] == jnew){new_point=0;}
                }
                if(new_point == 1)
                {
                    b[n][0] = r;
                    b[n][1] = c;
                    n++;
                }
                break;
            }
        }
    }

    for(int i=n;i<2*L;i++)
    {
        b[i][0] = 2*L;
        b[i][1] = 2*L;
    }

    double h_sum = 0, hsq_sum = 0;
    for (int i = 0; i < n; i++)
    {
        if (b[i][0]%2==0)
        {
            c = b[i][1];
        }
        else
        {
            c = b[i][1];
        }
        hsq_sum = hsq_sum + b[i][1]*b[i][1];
        h_sum = h_sum + b[i][1];
    }
    double h_ave = h_sum/n;
    double hsq_ave = hsq_sum/n;
    double w = sqrt(hsq_ave-h_ave*h_ave);
    boundary_width[t] = w;

}
