#pragma once
#include <iostream>
using std::cout;
using std::ostream;
#include <assert.h>
#include <string.h>
#include <math.h>

template<typename T,int n>
struct vec{
	static_assert(n>0,"");
	T data[n];
	constexpr __device__ vec(){for(int i=0;i<n;i++) data[i]=0;}
	constexpr __device__ vec(T x){for(int i=0;i<n;i++) data[i]=x;}
	constexpr __device__ vec(T x,T y){
		static_assert(n==2,"");
		data[0]=x,data[1]=y;
	}
	constexpr __device__ vec(T x,T y,T z){
		static_assert(n==3,"");
		data[0]=x,data[1]=y,data[2]=z;
	}
	constexpr __device__ vec(T x,T y,T z,T w){
		static_assert(n==4,"");
		data[0]=x,data[1]=y,data[2]=z,data[3]=w;
	}
	constexpr __device__ T &operator[](int i){return data[i];}
};


#define ELEMENTWISE(op,mod_op,ref)								\
template<typename T,int n>										\
constexpr __device__ vec<T,n> operator op(vec<T,n> ref a,vec<T,n> b){		\
	for(int i=0;i<n;i++){										\
		a[i] mod_op b[i];										\
	}															\
	return a;													\
}

//define element-wise operators
ELEMENTWISE(+ ,+=, );
ELEMENTWISE(- ,-=, );
ELEMENTWISE(* ,*=, );
ELEMENTWISE(/ ,/=, );
//define modifying element-wise operators
ELEMENTWISE(+=,+=,&);
ELEMENTWISE(-=,-=,&);
ELEMENTWISE(*=,*=,&);
ELEMENTWISE(/=,/=,&);

#undef ELEMENTWISE

#define SCALAR_ELEMENTWISE(op,mod_op,ref)			\
template<typename T,int n>							\
__device__ constexpr __device__ vec<T,n> operator op(vec<T,n> ref a,T x){	\
	for(int i=0;i<n;i++){							\
		a[i] mod_op x;								\
	}												\
	return a;										\
}

//define element-wise operators
SCALAR_ELEMENTWISE(* ,*=, );
SCALAR_ELEMENTWISE(/ ,/=, );
SCALAR_ELEMENTWISE(+ ,+=, );
SCALAR_ELEMENTWISE(- ,-=, );
//define modifying element-wise operators
SCALAR_ELEMENTWISE(*=,*=,&);
SCALAR_ELEMENTWISE(/=,/=,&);
SCALAR_ELEMENTWISE(+=,+=,&);
SCALAR_ELEMENTWISE(-=,-=,&);

#undef SCALAR_ELEMENTWISE


//for scalar first, vector second
#define SCALAR_ELEMENTWISE(op)						\
template<typename T,int n>							\
__device__ constexpr vec<T,n> operator op(T x,vec<T,n> a){		\
	for(int i=0;i<n;i++){							\
		a[i]=x op a[i];								\
	}												\
	return a;										\
}



//define element-wise operators
SCALAR_ELEMENTWISE(+);
SCALAR_ELEMENTWISE(-);
SCALAR_ELEMENTWISE(*);
SCALAR_ELEMENTWISE(/);

#undef SCALAR_ELEMENTWISE

template<typename T,int n>
__device__ constexpr T dot(vec<T,n> a,vec<T,n> b){
	T x=0;
	for(int i=0;i<n;i++) x+=a[i]*b[i];
	return x;
}

template<typename T>
__device__ constexpr vec<T,3> cross(vec<T,3> a,vec<T,3> b){
	vec<T,3> c;
	for(int i=0;i<3;i++){
		c[i]+=a[(i+1)%3]*b[(i+2)%3];
		c[i]-=a[(i-1+3)%3]*b[(i-2+3)%3];
	}
	return c;
}

template<typename T,int n>
__device__ constexpr T length(vec<T,n> a){
	return sqrtf(dot(a,a));
}

template<typename T,int n>
__device__ constexpr vec<T,n> normalize(vec<T,n> a){
	return a/length(a);
}

template<typename T,int n> 
__device__ constexpr bool operator==(vec<T,n> x,vec<T,n> y){
	for(int i=0;i<n;i++){ 
		if(x[i]!=y[i]) return false;
	}
	return true;
}

template<typename T,int n>
constexpr ostream &operator<<(ostream &os,vec<T,n> v){
	os<<"[";
	for(int i=0;i<n;i++){
		os<<v[i];
		if(i<n-1) cout<<", ";
	}
	os<<"]\n";
	return os;
}

template<typename T,int n>
__device__ constexpr vec<T,n> operator-(vec<T,n> v){
	return v*(T)-1;
}

template<typename T,int n>
__device__ constexpr vec<T,n> operator+(vec<T,n> v){
	return v;
}


//create nicer names for small vectors
typedef vec<float,2> vec2;
typedef vec<float,3> vec3;
typedef vec<float,4> vec4;





template<typename T,int r,int c>
struct matrix{
	static_assert(r>0 && c>0,"");
	T data[r][c];
	__device__ constexpr matrix(){
		if(r==c){
			for(int i=0;i<r;i++){
				for(int j=0;j<c;j++){
					data[i][j]=(i==j)?1:0;
				}
			}
		}
	}
	__device__ constexpr matrix(T x){
		static_assert(c==r,"");
		for(int i=0;i<r;i++){
			for(int j=0;j<c;j++){
				data[i][j]=(i==j)?x:0;
			}
		}
	}
	__device__ constexpr matrix(vec<T,r> v){
		static_assert(c==r,"");
		for(int i=0;i<r;i++){
			for(int j=0;j<c;j++){
				data[i][j]=(i==j)?v[i]:0;
			}
		}
	}
	__device__ constexpr matrix(vec<T,c> x,vec<T,c> y){
		static_assert(r==2,"");
		memcpy(data[0],x.data,c*sizeof(T));
		memcpy(data[1],y.data,c*sizeof(T));
	}
	__device__ constexpr matrix(vec<T,c> x,vec<T,c> y,vec<T,c> z){
		static_assert(r==3,"");
		memcpy(data[0],x.data,c*sizeof(T));
		memcpy(data[1],y.data,c*sizeof(T));
		memcpy(data[2],z.data,c*sizeof(T));
	}
	__device__ constexpr matrix(vec<T,c> x,vec<T,c> y,vec<T,c> z,vec<T,c> w){
		static_assert(r==4,"");
		memcpy(data[0],x.data,c*sizeof(T));
		memcpy(data[1],y.data,c*sizeof(T));
		memcpy(data[2],z.data,c*sizeof(T));
		memcpy(data[3],w.data,c*sizeof(T));
	}
	__device__  T *operator[](int i){
		assert(i<r);
		return data[i];
	}
};


//create nicer names for small matrices
typedef matrix<float,1,1> float1x1;
typedef matrix<float,1,2> float1x2;
typedef matrix<float,1,3> float1x3;
typedef matrix<float,1,4> float1x4;

typedef matrix<float,2,1> float2x1;
typedef matrix<float,2,2> float2x2;
typedef matrix<float,2,3> float2x3;
typedef matrix<float,2,4> float2x4;

typedef matrix<float,3,1> float3x1;
typedef matrix<float,3,2> float3x2;
typedef matrix<float,3,3> float3x3;
typedef matrix<float,3,4> float3x4;

typedef matrix<float,4,1> float4x1;
typedef matrix<float,4,2> float4x2;
typedef matrix<float,4,3> float4x3;
typedef matrix<float,4,4> float4x4;


template<typename T,int r,int c>
constexpr ostream &operator<<(ostream &os,matrix<T,r,c> m){
	os<<"[\n";
	for(int i=0;i<r;i++){
		os<<"\t[";
		for(int j=0;j<c;j++){
			os<<m.data[i][j];
			if(j<c-1) os<<", ";
		}
		os<<"]";
		if(i<r-1) cout<<",";
		os<<"\n";
	}
	os<<"]\n";
	return os;
}

#define ELEMENTWISE(op,mod_op,ref)											\
template<typename T,int r,int c>											\
__device__ constexpr matrix<T,r,c> operator op(matrix<T,r,c> ref a,matrix<T,r,c> b){	\
	for(int i=0;i<r;i++){													\
		for(int j=0;j<c;j++){												\
			a[i][j] mod_op b[i][j];											\
		}																	\
	}																		\
	return a;																\
}

//define element-wise operators
ELEMENTWISE(+ ,+=, );
ELEMENTWISE(- ,-=, );
//define modifying element-wise operators
ELEMENTWISE(+=,+=,&);
ELEMENTWISE(-=,-=,&);

#undef ELEMENTWISE



template<typename T,int r,int c> 
__device__ constexpr matrix<T,r,r> operator*(matrix<T,r,c> x,matrix<T,c,r> y){ 
	matrix<T,r,r> z;
	for(int i=0;i<r;i++){ 
		for(int j=0;j<r;j++){ 
			z[i][j]=0;
			for(int k=0;k<c;k++){ 
				z[i][j]+=x[i][k]*y[k][j]; 
			}
		} 
	} 
	return z; 
}

template<typename T,int n> 
__device__ constexpr matrix<T,n,n> operator*=(matrix<T,n,n> &x,matrix<T,n,n> y){
	return x=x*y;
}

template<typename T,int r,int c> 
__device__ constexpr matrix<T,c,r> transpose(matrix<T,r,c> x){
	matrix<T,c,r> t;
	for(int i=0;i<r;i++){ 
		for(int j=0;j<c;j++){ 
			t[j][i]=x[i][j];
		}
	}
	return t;
}


template<typename T,int n> 
__device__ constexpr matrix<T,n,n> inverse(matrix<T,n,n> x){
	//initialize data structure of 2 matrices side by side
	//the first is x and the second is the identity matrix
	T data_raw[n][2*n];
	T *data[n];
	for(int i=0;i<n;i++) data[i]=&data_raw[i][0];
	for(int i=0;i<n;i++){ 
		for(int j=0;j<2*n;j++){
			if(j<n) data[i][j]=x[i][j];
			else data[i][j]=(i==j-n)?1:0;
		}
	}
	//make the first matrix diagonal
	for(int i=0;i<n;i++){
        if(data[i][i]==0){
            bool b=false;
            for(int j=i+1;j<n;j++){
                if(data[j][i]!=0){
                    T *tmp=data[i];
                    data[i]=data[j];
                    data[j]=tmp;
                    b=true;
                    break;
                }
            }
        }
        for(int j=i+1;j<n;j++){
        	T coef=-data[j][i]/data[i][i];
            for(int k=0;k<2*n;k++) data[j][k]+=coef*data[i][k];
		}
	}
	//make the first matrix the identity matrix 
	for(int i=n-1;i>0;i--){
        if(data[i][i]==0){
            bool b=false;
            for(int j=n-1;j>i;j--){
                if(data[j][i]!=0){
                    T *tmp=data[i];
                    data[i]=data[j];
                    data[j]=tmp;
                    b=true;
                    break;
                }
            }
        }
        for(int j=i-1;j>=0;j--){
            T coef=-data[j][i]/data[i][i];
            for(int k=0;k<2*n;k++) data[j][k]+=coef*data[i][k];
        }
    }
    for(int i=0;i<n;i++){
		for(int j=2*n-1;j>=n;j--){
            data[i][j]/=data[i][i];
		}
    }
    //the second matrix is now the inverse(x)
    //copy the second matrix into inv and return it
	matrix<T,n,n> inv;
	for(int i=0;i<n;i++){ 
		for(int j=0;j<n;j++){
			inv[i][j]=data[i][j+n];
		}
	} 
	return inv;
}

template<typename T,int n> 
__device__ constexpr matrix<T,n,n> operator/(matrix<T,n,n> x,matrix<T,n,n> y){
	return x*inverse(y);
}

template<typename T,int n> 
__device__ constexpr matrix<T,n,n> operator/=(matrix<T,n,n> &x,matrix<T,n,n> y){
	return x*=inverse(y);
}

template<typename T,int n>
__device__ constexpr matrix<T,n,n> pow(matrix<T,n,n> m,int p){
	matrix<T,n,n> x;
	if(p>=0){
		for(int i=0;i<p;i++) x*=m;
		return x;
	}
	p=abs(p);
	auto m_inv=inverse(m);
	for(int i=0;i<p;i++) x*=m_inv;
	return x;
}

template<typename T,int r,int c> 
__device__ constexpr bool operator==(matrix<T,r,c> x,matrix<T,r,c> y){
	for(int i=0;i<r;i++){ 
		for(int j=0;j<c;j++){ 
			if(x[i][j]!=y[i][j]) return false;
		}
	}
	return true;
}


//for matrix first, scalar second
#define SCALAR_ELEMENTWISE(op,mod_op,ref)			\
template<typename T,int r,int c>					\
__device__ constexpr matrix<T,r,c> operator op(matrix<T,r,c> ref a,T x){	\
	for(int i=0;i<r;i++){							\
		for(int j=0;j<c;j++){						\
			a[i][j] mod_op x;						\
		}											\
	}												\
	return a;										\
}

//define element-wise operators
SCALAR_ELEMENTWISE(* ,*=, );
SCALAR_ELEMENTWISE(/ ,/=, );
//define modifying element-wise operators
SCALAR_ELEMENTWISE(*=,*=,&);
SCALAR_ELEMENTWISE(/=,/=,&);

#undef SCALAR_ELEMENTWISE


//for scalar first, matrix second
#define SCALAR_ELEMENTWISE(op)					\
template<typename T,int r,int c>				\
__device__ constexpr matrix<T,r,c> operator op(T x,matrix<T,r,c> a){	\
	for(int i=0;i<r;i++){						\
		for(int j=0;j<c;j++){					\
			a[i][j]=x op a[i][j];				\
		}										\
	}											\
	return a;									\
}

//define element-wise operators
SCALAR_ELEMENTWISE(*);
SCALAR_ELEMENTWISE(/);

#undef SCALAR_ELEMENTWISE


//matrix-vector multiplication
template<typename T,int r,int c>
__device__ constexpr vec<T,r> operator*(matrix<T,r,c> m,vec<T,c> v){
	vec<T,r> u;
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			u[i]+=v[j]*m[i][j];
		}
	}
	return u;
}

template<typename T,int r,int c>
__device__ constexpr vec<T,c> operator*(vec<T,r> v,matrix<T,r,c> m){
	vec<T,c> u;
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			u[j]+=v[i]*m[i][j];
		}
	}
	return u;
}


template<typename T,int n>
__device__  matrix<T,n+1,n+1> translate(vec<T,n> v){
	matrix<T,n+1,n+1> m;
	for(int i=0;i<n;i++) m[i][n]=v[i];
	return m;
}

template<typename T,int n>
__device__ constexpr vec<T,n> mv(matrix<T,n+1,n+1> m,vec<T,n> v,bool tr){
	vec<T,n+1> vn;
	memcpy(&vn[0],&v[0],n*sizeof(T));
	vn[n]=(T)tr;
	auto vx=m*vn;
	memcpy(&v[0],&vx[0],n*sizeof(T));
	return v;
}


/*
rotate<T,2>(theta,0b0000); //one way to rotate in 2D 
rotate<T,3>(theta,0b0001); //about x-axis
rotate<T,3>(theta,0b0010); //about y-axis
rotate<T,3>(theta,0b0100); //about z-axis
rotate<T,4>(theta,0b0011); //about xy-plane
rotate<T,4>(theta,0b0101); //about xz-plane
rotate<T,4>(theta,0b0110); //about yz-plane
rotate<T,4>(theta,0b1001); //about xw-plane
rotate<T,4>(theta,0b1010); //about yw-plane
rotate<T,4>(theta,0b1100); //about zw-plane
*/

template<typename T>
__device__ constexpr T popcount(T x){
	T pc=0;
	for(int i=0;i<8*sizeof(T);i++) if((1<<i)&x) pc++;
	return pc;
}

template<typename T,int n>
__device__ constexpr matrix<T,n,n> rotate(T theta,unsigned int enc){
	static_assert(n>=2 && n<32,"");
	assert(popcount(enc)==n-2);
	T s,c;
	s=sinf(theta),c=cosf(theta);
	// else s=sin(theta),c=cos(theta);
	matrix<T,n,n> m;
	int j=0,idxs[2]={0,1};
	for(int i=0;i<n;i++){
		if(!(enc&(1<<i))){
			idxs[j]=i;
			j++;
		}
	}
	m[idxs[0]][idxs[0]]=+c;
	m[idxs[1]][idxs[1]]=+c;
	m[idxs[0]][idxs[1]]=-s;
	m[idxs[1]][idxs[0]]=+s;
	return m;
}

template<typename T,int n>
__device__ vec<T,n> abs(vec<T,n> a){
	for(int i=0;i<n;i++) a[i]=(a[i]>0)?a[i]:-a[i];
	return a;
}


template<typename T,int n>
__device__ vec<T,n> max(vec<T,n> a,vec<T,n> b){
	for(int i=0;i<n;i++) a[i]=(a[i]>b[i])?a[i]:b[i];
	return a;
}

template<typename T,int n>
__device__ vec<T,n> min(vec<T,n> a,vec<T,n> b){
	for(int i=0;i<n;i++) a[i]=(a[i]<b[i])?a[i]:b[i];
	return a;
}

template<typename T,int n>
__device__ vec<T,n> clamp(vec<T,n> a,T lo,T hi){
	return min(max(a,vec<T,n>(lo)),vec<T,n>(hi));
}


































//

