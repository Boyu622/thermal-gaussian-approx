#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define RADIX 2.0

/*************************
 * balance a real matrix *
 *************************/

#include <stdio.h>
#include <math.h>

void balanc(double **a, int n)
{
    int             i, j, last = 0;
    double          s, r, g, f, c, sqrdx;

    sqrdx = RADIX * RADIX;
    while (last == 0) {
        last = 1;
        for (i = 0; i < n; i++) {
            r = c = 0.0;
            for (j = 0; j < n; j++)
                if (j != i) {
                    c += fabs(a[j][i]);
                    r += fabs(a[i][j]);
                }
            if (c != 0.0 && r != 0.0) {
                g = r / RADIX;
                f = 1.0;
                s = c + r;
                while (c < g) {
                    f *= RADIX;
                    c *= sqrdx;
                }
                g = r * RADIX;
                while (c > g) {
                    f /= RADIX;
                    c /= sqrdx;
                }
                if ((c + r) / f < 0.95 * s) {
                    last = 0;
                    g = 1.0 / f;
                    for (j = 0; j < n; j++)
                        a[i][j] *= g;
                    for (j = 0; j < n; j++)
                        a[j][i] *= f;
                }
            }
        }
    }
}

#define SWAP(a,b) do { double t = (a); (a) = (b); (b) = t; } while (0)

/*****************************************************
 * convert a non-symmetric matrix to Hessenberg form *
 *****************************************************/

void elmhes(double **a, int n)
{
    int             i, j, m;
    double          y, x;

    for (m = 1; m < n - 1; m++) {
        x = 0.0;
        i = m;
        for (j = m; j < n; j++) {
            if (fabs(a[j][m - 1]) > fabs(x)) {
                x = a[j][m - 1];
                i = j;
            }
        }
        if (i != m) {
            for (j = m - 1; j < n; j++)
                SWAP(a[i][j], a[m][j]);
            for (j = 0; j < n; j++)
                SWAP(a[j][i], a[j][m]);
        }
        if (x != 0.0) {
            for (i = m + 1; i < n; i++) {
                if ((y = a[i][m - 1]) != 0.0) {
                    y /= x;
                    a[i][m - 1] = y;
                    for (j = m; j < n; j++)
                        a[i][j] -= y * a[m][j];
                    for (j = 0; j < n; j++)
                        a[j][m] += y * a[j][i];
                }
            }
        }
    }
}

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

/**************************************
 * QR algorithm for Hessenberg matrix *
 **************************************/

void hqr(double **a, int n, double *wr, double *wi)
{
    int             nn, m, l, k, j, its, i, mmin;
    double          z, y, x, w, v, u, t, s, r, q, p, anorm;

    p = q = r = 0.0;
    anorm = 0.0;
    for (i = 0; i < n; i++)
        for (j = i - 1 > 0 ? i - 1 : 0; j < n; j++)
            anorm += fabs(a[i][j]);
    nn = n - 1;
    t = 0.0;
    while (nn >= 0) {
        its = 0;
        do {
            for (l = nn; l > 0; l--) {
                s = fabs(a[l - 1][l - 1]) + fabs(a[l][l]);
                if (s == 0.0)
                    s = anorm;
                if (fabs(a[l][l - 1]) + s == s) {
                    a[l][l - 1] = 0.0;
                    break;
                }
            }
            x = a[nn][nn];
            if (l == nn) {
                wr[nn] = x + t;
                wi[nn--] = 0.0;
            } else {
                y = a[nn - 1][nn - 1];
                w = a[nn][nn - 1] * a[nn - 1][nn];
                if (l == nn - 1) {
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    z = sqrt(fabs(q));
                    x += t;
                    if (q >= 0.0) {
                        z = p + SIGN(z, p);
                        wr[nn - 1] = wr[nn] = x + z;
                        if (z != 0.0)
                            wr[nn] = x - w / z;
                        wi[nn - 1] = wi[nn] = 0.0;
                    } else {
                        wr[nn - 1] = wr[nn] = x + p;
                        wi[nn - 1] = -(wi[nn] = z);
                    }
                    nn -= 2;
                } else {
                    if (its == 30) {
                        fprintf(stderr, "[hqr] too many iterations.\n");
                        break;
                    }
                    if (its == 10 || its == 20) {
                        t += x;
                        for (i = 0; i < nn + 1; i++)
                            a[i][i] -= x;
                        s = fabs(a[nn][nn - 1]) + fabs(a[nn - 1][nn - 2]);
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    for (m = nn - 2; m >= l; m--) {
                        z = a[m][m];
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
                        q = a[m + 1][m + 1] - z - r - s;
                        r = a[m + 2][m + 1];
                        s = fabs(p) + fabs(q) + fabs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l)
                            break;
                        u = fabs(a[m][m - 1]) * (fabs(q) + fabs(r));
                        v = fabs(p) * (fabs(a[m - 1][m - 1]) + fabs(z) + fabs(a[m + 1][m + 1]));
                        if (u + v == v)
                            break;
                    }
                    for (i = m; i < nn - 1; i++) {
                        a[i + 2][i] = 0.0;
                        if (i != m)
                            a[i + 2][i - 1] = 0.0;
                    }
                    for (k = m; k < nn; k++) {
                        if (k != m) {
                            p = a[k][k - 1];
                            q = a[k + 1][k - 1];
                            r = 0.0;
                            if (k + 1 != nn)
                                r = a[k + 2][k - 1];
                            if ((x = fabs(p) + fabs(q) + fabs(r)) != 0.0) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        if ((s = SIGN(sqrt(p * p + q * q + r * r), p)) != 0.0) {
                            if (k == m) {
                                if (l != m)
                                    a[k][k - 1] = -a[k][k - 1];
                            } else
                                a[k][k - 1] = -s * x;
                            p += s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q /= p;
                            r /= p;
                            for (j = k; j < nn + 1; j++) {
                                p = a[k][j] + q * a[k + 1][j];
                                if (k + 1 != nn) {
                                    p += r * a[k + 2][j];
                                    a[k + 2][j] -= p * z;
                                }
                                a[k + 1][j] -= p * y;
                                a[k][j] -= p * x;
                            }
                            mmin = nn < k + 3 ? nn : k + 3;
                            for (i = l; i < mmin + 1; i++) {
                                p = x * a[i][k] + y * a[i][k + 1];
                                if (k != (nn)) {
                                    p += z * a[i][k + 2];
                                    a[i][k + 2] -= p * r;
                                }
                                a[i][k + 1] -= p * q;
                                a[i][k] -= p;
                            }
                        }
                    }
                }
            }
        } while (l + 1 < nn);
    }
}

/*********************************************************
 * calculate eigenvalues for a non-symmetric real matrix *
 *********************************************************/

void n_eigen(double *_a, int n, double *wr, double *wi)
{
    int             i;
    double        **a = (double **) calloc(n, sizeof(void *));
    for (i = 0; i < n; ++i)
        a[i] = _a + i * n;
    balanc(a, n);
    elmhes(a, n);
    hqr(a, n, wr, wi);
    free(a);
}

/* convert a symmetric matrix to tridiagonal form */

#define SQR(a) ((a)*(a))

double pythag(double a, double b)
{
    double absa, absb;
    absa = fabs(a);
    absb = fabs(b);
    if (absa > absb) return absa * sqrt(1.0 + SQR(absb / absa));
    else return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

void tred2(double **a, int n, double *d, double *e)
{
    int             l, k, j, i;
    double          scale, hh, h, g, f;

    for (i = n - 1; i > 0; i--) {
        l = i - 1;
        h = scale = 0.0;
        if (l > 0) {
            for (k = 0; k < l + 1; k++)
                scale += fabs(a[i][k]);
            if (scale == 0.0)
                e[i] = a[i][l];
            else {
                for (k = 0; k < l + 1; k++) {
                    a[i][k] /= scale;
                    h += a[i][k] * a[i][k];
                }
                f = a[i][l];
                g = (f >= 0.0 ? -sqrt(h) : sqrt(h));
                e[i] = scale * g;
                h -= f * g;
                a[i][l] = f - g;
                f = 0.0;
                for (j = 0; j < l + 1; j++) {
                    /* Next statement can be omitted if eigenvectors not wanted */
                    a[j][i] = a[i][j] / h;
                    g = 0.0;
                    for (k = 0; k < j + 1; k++)
                        g += a[j][k] * a[i][k];
                    for (k = j + 1; k < l + 1; k++)
                        g += a[k][j] * a[i][k];
                    e[j] = g / h;
                    f += e[j] * a[i][j];
                }
                hh = f / (h + h);
                for (j = 0; j < l + 1; j++) {
                    f = a[i][j];
                    e[j] = g = e[j] - hh * f;
                    for (k = 0; k < j + 1; k++)
                        a[j][k] -= (f * e[k] + g * a[i][k]);
                }
            }
        } else
            e[i] = a[i][l];
        d[i] = h;
    }
    /* Next statement can be omitted if eigenvectors not wanted */
    d[0] = 0.0;
    e[0] = 0.0;
    /* Contents of this loop can be omitted if eigenvectors not wanted except for statement d[i]=a[i][i]; */
    for (i = 0; i < n; i++) {
        l = i;
        if (d[i] != 0.0) {
            for (j = 0; j < l; j++) {
                g = 0.0;
                for (k = 0; k < l; k++)
                    g += a[i][k] * a[k][j];
                for (k = 0; k < l; k++)
                    a[k][j] -= g * a[k][i];
            }
        }
        d[i] = a[i][i];
        a[i][i] = 1.0;
        for (j = 0; j < l; j++)
            a[j][i] = a[i][j] = 0.0;
    }
}

/* calculate the eigenvalues and eigenvectors of a symmetric tridiagonal matrix */
void tqli(double *d, double *e, int n, double **z)
{
    int             m, l, iter, i, k;
    double          s, r, p, g, f, dd, c, b;

    for (i = 1; i < n; i++)
        e[i - 1] = e[i];
    e[n - 1] = 0.0;
    for (l = 0; l < n; l++) {
        iter = 0;
        do {
            for (m = l; m < n - 1; m++) {
                dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + dd == dd)
                    break;
            }
            if (m != l) {
                if (iter++ == 30) {
                    fprintf(stderr, "[tqli] Too many iterations in tqli.\n");
                    break;
                }
                g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                r = pythag(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
                s = c = 1.0;
                p = 0.0;
                for (i = m - 1; i >= l; i--) {
                    f = s * e[i];
                    b = c * e[i];
                    e[i + 1] = (r = pythag(f, g));
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    /* Next loop can be omitted if eigenvectors not wanted */
                    for (k = 0; k < n; k++) {
                        f = z[k][i + 1];
                        z[k][i + 1] = s * z[k][i] + c * f;
                        z[k][i] = c * z[k][i] - s * f;
                    }
                }
                if (r == 0.0 && i >= l)
                    continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
}

int n_eigen_symm(double *_a, int n, double *eval)
{
    double **a, *e;
    int i;
    a = (double**)calloc(n, sizeof(void*));
    e = (double*)calloc(n, sizeof(double));
    for (i = 0; i < n; ++i) a[i] = _a + i * n;
    tred2(a, n, eval, e);
    tqli(eval, e, n, a);
    free(a); free(e);
    return 0;
}

int main(void)
{
    double lambda1[3][3];
    double lambda2[3][3];
    double lambda3[3][3];
    double lambda4[3][3];
    double lambda5[3][3];
    double identity[3][3];
    //lambda1
    lambda1[0][0]=0.0;
    lambda1[0][1]=1.0;
    lambda1[0][2]=0.0;
    
    lambda1[1][0]=1.0;
    lambda1[1][1]=0.0;
    lambda1[1][2]=0.0;
    
    lambda1[2][0]=0.0;
    lambda1[2][1]=0.0;
    lambda1[2][2]=0.0;
    
    //lambda2
    lambda2[0][0]=1.0;
    lambda2[0][1]=0.0;
    lambda2[0][2]=0.0;
    
    lambda2[1][0]=0.0;
    lambda2[1][1]=-1.0;
    lambda2[1][2]=0.0;
    
    lambda2[2][0]=0.0;
    lambda2[2][1]=0.0;
    lambda2[2][2]=0.0;
    
    //lambda3
    lambda3[0][0]=0.0;
    lambda3[0][1]=0.0;
    lambda3[0][2]=1.0;
    
    lambda3[1][0]=0.0;
    lambda3[1][1]=0.0;
    lambda3[1][2]=0.0;
    
    lambda3[2][0]=1.0;
    lambda3[2][1]=0.0;
    lambda3[2][2]=0.0;
    
    //lambda4
    lambda4[0][0]=0.0;
    lambda4[0][1]=0.0;
    lambda4[0][2]=0.0;
    
    lambda4[1][0]=0.0;
    lambda4[1][1]=0.0;
    lambda4[1][2]=1.0;
    
    lambda4[2][0]=0.0;
    lambda4[2][1]=1.0;
    lambda4[2][2]=0.0;
    
    //lambda5
    lambda5[0][0]=1.0/sqrt(3);
    lambda5[0][1]=0.0;
    lambda5[0][2]=0.0;
    
    lambda5[1][0]=0.0;
    lambda5[1][1]=1.0/sqrt(3);
    lambda5[1][2]=0.0;
    
    lambda5[2][0]=0.0;
    lambda5[2][1]=0.0;
    lambda5[2][2]=-2/sqrt(3);
    
    //identity
    identity[0][0]=1.0;
    identity[0][1]=0.0;
    identity[0][2]=0.0;
    
    identity[1][0]=0.0;
    identity[1][1]=1.0;
    identity[1][2]=0.0;
    
    identity[2][0]=0.0;
    identity[2][1]=0.0;
    identity[2][2]=1.0;
    
    double mhinv[3][3];
    double nhinv[3][3];
    //MH^-1
    mhinv[0][0]=0.92045186228036213046901110967998426133125987738094170356015852509611669275517;
    mhinv[0][1]=0.368238512946166998293513187385268219366950034802393343078114695767650134476778;
    mhinv[0][2]=0.184154054602409827340641632873685049525816453213736392977413336044012490248841;
    
    mhinv[1][0]=0.368238512946166998293513187385268219366950034802393343078114695767650134476778;
    mhinv[1][1]=1.03094409321855536688972627708440132333439815566711229002695413326518755088588;
    mhinv[1][2]=0.368255525364948758759750593248773661324566737670333373126334773129400047372444;
    
    mhinv[2][0]=0.184154054602409827340641632873685049525816453213736392977413336044012490248841;
    mhinv[2][1]=0.368255525364948758759750593248773661324566737670333373126334773129400047372444;
    mhinv[2][2]=0.92048046527554855271122258466096970462810584736900918191416273582039732001899;
    
    //NH^-1
    nhinv[0][0]=0.00491596331949039175875389862888142346440214443623901258953640058687525215861512;
    nhinv[0][1]=0.00641872567683048716257023475610149343292474840047309147021932652406494079202064;
    nhinv[0][2]=0.00586604348418603924981606675039143217705012822167234052300451482713511462075708;
    
    nhinv[1][0]=0.00641872567683048716257023475610149343292474840047309147021932652406494079202064;
    nhinv[1][1]=0.0084099118052130528061694657548479802508920863063750560632136518628868122050802;
    nhinv[1][2]=0.0077175473258739872814028600286375629867858050020978548428095325919462546213723;
    
    nhinv[2][0]=0.00586604348418603924981606675039143217705012822167234052300451482713511462075708;
    nhinv[2][1]=0.0077175473258739872814028600286375629867858050020978548428095325919462546213723;
    nhinv[2][2]=0.00711758757655913161445889135814281324487116638684935379242890493667108206432752;
    
    
    double uppersum1=13.555181569708505574692401556257910589633900793816094129946402465618023346093792;
    double uppersum2=0.03042113399574585829671277432227951201844499582825020161855370323411408884844;
    double uppersum3=13.555291239993434340160191567535650534954810256489649042614639973291850873786495;
    double uppersum4=13.611860083434000352898339591691951336429744579164483446468972049389981991594722;
    double uppersum5=-0.08380605730849019811381234517429688290875723473011859080325907829846632802555;
    double uppersum6=13.63359027995556078892953456317944081421175927938342901738000444268783264499852;
    
    double uppersum[3][3];
    for(int i=0;i<=2;i++){
        for(int j=0;j<=2;j++){
            uppersum[i][j]=uppersum1*lambda1[i][j]+uppersum2*lambda2[i][j]+uppersum3*lambda3[i][j]+uppersum4*lambda4[i][j]+uppersum5*lambda5[i][j]+uppersum6*identity[i][j];
    }
    }
    
    
    double upperminus1=0.417627438168462097240101295743503978190032642050993668362317723794647055765245;
    double upperminus2=0.08007109694244737812400239308213754998469328245792038484824190493495676928227;
    double upperminus3=0.361273502992660111738813111130954654415226730122407965073357554859174721153025;
    double upperminus4=0.360914899605403798101688448582452347478955450966724291743307985299188584472983;
    double upperminus5=0.02004636592690305783139124233870924010764185659933164807065316662677188811310;
    double upperminus6=0.43435101803259286565526425039158654203144417373801182660013688797081972457268;
    
    double upperminus[3][3];
    for(int i=0;i<=2;i++){
        for(int j=0;j<=2;j++){
            upperminus[i][j]=upperminus1*lambda1[i][j]+upperminus2*lambda2[i][j]+upperminus3*lambda3[i][j]+upperminus4*lambda4[i][j]+upperminus5*lambda5[i][j]+upperminus6*identity[i][j];
    }
    }
    
//    get pool of matrices smaller than m+n, and store them in validmatrixmpn
    double validmatrixmpn[3][3][10000];
    for(int i=0;i<=2;i++){
        for(int j=0;j<=2;j++){
            validmatrixmpn[i][j][0]=0.0;
    }
    }
    
    int pindicator = 1;
    for (double ppq1=-0.1; ppq1<=0.1; ppq1=ppq1+0.001){
        printf("ppq1:");
        printf("%.6f", ppq1);
        printf("\n");
        for (double ppq2=-0.1; ppq2<=0.1; ppq2=ppq2+0.001){

            for (double ppq3=-0.1; ppq3<=0.1; ppq3=ppq3+0.001){
                
                for (double ppq4=-0.1; ppq4<=0.1; ppq4=ppq4+0.001){

                    for (double ppq5=-0.1; ppq5<=0.1; ppq5=ppq5+0.001){

                        for (double ppq6=-0.1; ppq6<=0.1; ppq6=ppq6+0.001){
                            //matrix p+q positive
                            double tsetppq[3][3];
                            for(int i=0;i<=2;i++){
                                for(int j=0;j<=2;j++){
                                    tsetppq[i][j]=ppq1*lambda1[i][j]+ppq2*lambda2[i][j]+ppq3*lambda3[i][j]+ppq4*lambda4[i][j]+ppq5*lambda5[i][j]+ppq6*identity[i][j];
                            }
                            }
                            
                            //matrix m+n-p-q positive
                            double tsetmpnmm[3][3];
                            for(int i=0;i<=2;i++){
                                for(int j=0;j<=2;j++){
                                    tsetmpnmm[i][j]=uppersum[i][j]-tsetppq[i][j];
                            }
                            }
                            
                            double   u[3];
                            u[0]=0.0;
                            u[1]=0.0;
                            u[2]=0.0;
                            n_eigen_symm(tsetppq[0], 3, u);
                            
                            double   v[3];
                            v[0]=0.0;
                            v[1]=0.0;
                            v[2]=0.0;
                            n_eigen_symm(tsetmpnmm[0], 3, v);

                            
                            
                            if(u[0]>=0.0&&u[1]>=0.0&&u[2]>=0.0&&v[0]>=0.0&&v[1]>=0.0&&v[2]>=0.0)
                            {

                                
                                for(int i=0;i<=2;i++){
                                    for(int j=0;j<=2;j++){
                                        validmatrixmpn[i][j][pindicator]=ppq1*lambda1[i][j]+ppq2*lambda2[i][j]+ppq3*lambda3[i][j]+ppq4*lambda4[i][j]+ppq5*lambda5[i][j]+ppq6*identity[i][j];
                                }
                                }
                                pindicator++;
                                
                            }

                            
                         
                                        }}}}}}
    
    //    get pool of matrices smaller than m-n, and store them in validmatrixmmn
    double validmatrixmmn[3][3][10000];
    for(int i=0;i<=2;i++){
        for(int j=0;j<=2;j++){
            validmatrixmmn[i][j][0]=0.0;
    }
    }
    int mindicator = 1;
    for (double pmq1=-0.01; pmq1<=0.01; pmq1=pmq1+0.0001){
        printf("pmq1:");
        printf("%.20f", pmq1);
        printf("\n");
        for (double pmq2=-0.01; pmq2<=0.01; pmq2=pmq2+0.0001){

            for (double pmq3=-0.01; pmq3<=0.01; pmq3=pmq3+0.0001){

                for (double pmq4=-0.01; pmq4<=0.01; pmq4=pmq4+0.0001){

                    for (double pmq5=-0.01; pmq5<=0.01; pmq5=pmq5+0.0001){

                        for (double pmq6=-0.01; pmq6<=0.01; pmq6=pmq6+0.0001){
                            //matrix p-q positive
                            double tsetpmq[3][3];
                            for(int i=0;i<=2;i++){
                                for(int j=0;j<=2;j++){
                                    tsetpmq[i][j]=pmq1*lambda1[i][j]+pmq2*lambda2[i][j]+pmq3*lambda3[i][j]+pmq4*lambda4[i][j]+pmq5*lambda5[i][j]+pmq6*identity[i][j];
                            }
                            }
                            
                            //matrix m-n-p+q positive
                            double tsetmmnmm[3][3];
                            for(int i=0;i<=2;i++){
                                for(int j=0;j<=2;j++){
                                    tsetmmnmm[i][j]=upperminus[i][j]-tsetpmq[i][j];
                            }
                            }
                            
                            double   o[3];
                            o[0]=0.0;
                            o[1]=0.0;
                            o[2]=0.0;
                            n_eigen_symm(tsetpmq[0], 3, o);
                            
                            double   d[3];
                            d[0]=0.0;
                            d[1]=0.0;
                            d[2]=0.0;
                            n_eigen_symm(tsetmmnmm[0], 3, d);
                            
                        
                            
                            if(o[0]>=0.0&&o[1]>=0.0&&o[2]>=0.0&&d[0]>=0.0&&d[1]>=0.0&&d[2]>=0.0)
                            {
                                for(int i=0;i<=2;i++){
                                    for(int j=0;j<=2;j++){
                                        validmatrixmmn[i][j][mindicator]=pmq1*lambda1[i][j]+pmq2*lambda2[i][j]+pmq3*lambda3[i][j]+pmq4*lambda4[i][j]+pmq5*lambda5[i][j]+pmq6*identity[i][j];
                                }
                                }
                                mindicator++;
                                
                            }

                            
                         
                                        }}}}}}
    
//  calculate and compare local determinant
    double vallocaldet[pindicator][mindicator];
    for(int k=0;k<=pindicator-1;k++){
        for(int l=0;l<=mindicator-1;l++){
            
            //           calculate numerator value
                        double numeratormatrix[3][3];
                        for(int i=0;i<=2;i++){
                            for(int j=0;j<=2;j++){
                                numeratormatrix[i][j]=mhinv[i][j]+(validmatrixmpn[i][j][k]+validmatrixmmn[i][j][l])/2;
                        }
                        }
                        double   o1[3];
                        o1[0]=0.0;
                        o1[1]=0.0;
                        o1[2]=0.0;
                        n_eigen_symm(numeratormatrix[0], 3, o1);
                        double valnumerator=o1[0]*o1[1]*o1[2];
            
            //           calculate denominator value
                        double denominatormatrix1[3][3];
                        double denominatormatrix2[3][3];
                        for(int i=0;i<=2;i++){
                            for(int j=0;j<=2;j++){
                                denominatormatrix1[i][j]=mhinv[i][j]+nhinv[i][j]+validmatrixmpn[i][j][k];
                        }
                        }
                        for(int i=0;i<=2;i++){
                            for(int j=0;j<=2;j++){
                                denominatormatrix2[i][j]=mhinv[i][j]-nhinv[i][j]+validmatrixmmn[i][j][l];
                        }
                        }
            
            double   o2[3];
            o2[0]=0.0;
            o2[1]=0.0;
            o2[2]=0.0;
            n_eigen_symm(denominatormatrix1[0], 3, o2);
            double valdenominator1=o2[0]*o2[1]*o2[2];
            
            double   o3[3];
            o3[0]=0.0;
            o3[1]=0.0;
            o3[2]=0.0;
            n_eigen_symm(denominatormatrix2[0], 3, o3);
            double valdenominator2=o3[0]*o3[1]*o3[2];
            
            vallocaldet[k][l]=valnumerator*valnumerator/(valdenominator1*valdenominator2);

            
    }
    }
    

    
    double minval=vallocaldet[0][0];
    int mink=0;
    int minl=0;
    for(int k=0;k<=pindicator-1;k++){
        for(int l=0;l<=mindicator-1;l++){
            
            if(vallocaldet[k][l]<minval){
                minval=vallocaldet[k][l];
                mink=k;
                minl=l;
            }
            
        }}
 
        printf("minval:");
        printf("%.6f", minval);
    
        printf("\n");
        printf("p+q:");
        printf("{");
        printf("{");
        printf("%.20f", validmatrixmpn[0][0][mink]);
        printf(",");
        printf("%.20f", validmatrixmpn[0][1][mink]);
        printf(",");
        printf("%.20f", validmatrixmpn[0][2][mink]);
        printf("}");
        printf(",");
        printf("{");
        printf("%.20f", validmatrixmpn[1][0][mink]);
        printf(",");
        printf("%.20f", validmatrixmpn[1][1][mink]);
        printf(",");
        printf("%.20f", validmatrixmpn[1][2][mink]);
        printf("}");
        printf(",");
        printf("{");
        printf("%.20f", validmatrixmpn[2][0][mink]);
        printf(",");
        printf("%.20f", validmatrixmpn[2][1][mink]);
        printf(",");
        printf("%.20f", validmatrixmpn[2][2][mink]);
        printf("}");
        printf("}");
    
        printf("\n");
        printf("p-q:");
        printf("{");
        printf("{");
        printf("%.20f", validmatrixmmn[0][0][minl]);
        printf(",");
        printf("%.20f", validmatrixmmn[0][1][minl]);
        printf(",");
        printf("%.20f", validmatrixmmn[0][2][minl]);
        printf("}");
        printf(",");
        printf("{");
        printf("%.20f", validmatrixmmn[1][0][minl]);
        printf(",");
        printf("%.20f", validmatrixmmn[1][1][minl]);
        printf(",");
        printf("%.20f", validmatrixmmn[1][2][minl]);
        printf("}");
        printf(",");
        printf("{");
        printf("%.20f", validmatrixmmn[2][0][minl]);
        printf(",");
        printf("%.20f", validmatrixmmn[2][1][minl]);
        printf(",");
        printf("%.20f", validmatrixmmn[2][2][minl]);
        printf("}");
        printf("}");
        printf("\n");
    
    
    
    
    }
