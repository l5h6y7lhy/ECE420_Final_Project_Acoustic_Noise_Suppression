#include "ece420_main.h"
#include "ece420_lib.h"
#include "kiss_fft/kiss_fft.h"
#include <algorithm>
#include <math.h>

// Student Variables
#define FRAME_SIZE 1024
#define F_S 48000
#define VOICED_THRESHOLD 990000000  // Find your own threshold

kiss_fft_cpx bufferIn[2*FRAME_SIZE];
kiss_fft_cpx bufferIn_w[FRAME_SIZE];
kiss_fft_cpx bufferIn_l[FRAME_SIZE];
kiss_fft_cpx in[2*FRAME_SIZE];
kiss_fft_cpx in1[2*FRAME_SIZE];
kiss_fft_cpx out[2*FRAME_SIZE];
kiss_fft_cpx in_w[FRAME_SIZE];
kiss_fft_cpx in_l[FRAME_SIZE];
kiss_fft_cpx out_w[FRAME_SIZE];
kiss_fft_cpx out_l[FRAME_SIZE];
float noise_spec[2*FRAME_SIZE];
float noise_spec_w[FRAME_SIZE];
float noise_spec_l[FRAME_SIZE];
float wiener[FRAME_SIZE];
float residual[2*FRAME_SIZE];
int16_t output_buff[FRAME_SIZE];

kiss_fft_cpx prev[2*FRAME_SIZE];
kiss_fft_cpx curr[2*FRAME_SIZE];

bool first_frame = true;
bool first_frame_w = true;
bool first_frame_l = true;
int curr_flag;

void ece420ProcessFrame(sample_buf *dataBuf) {
// Keep in mind, we only have 20ms to process each buffer!
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    long long energy=0;
    kiss_fft_cpx transition;
    int flag=1;

    if(first_frame) {
        for(int i=0; i<2*FRAME_SIZE; i++) {residual[i] = 0.0;}
    }

    // Data is encoded in signed PCM-16, little-endian, mono
    for (int i = 0; i < FRAME_SIZE; i++) {
        int16_t val = ((uint16_t) dataBuf->buf_[2 * i]) | (((uint16_t) dataBuf->buf_[2 * i + 1]) << 8);

        bufferIn[i].r = bufferIn[1024+i].r;
        bufferIn[i].i = 0;
        bufferIn[1024+i].r = ((float) val);
        bufferIn[1024+i].i = 0;
        energy = energy + bufferIn[i].r*bufferIn[i].r + bufferIn[1024+i].r*bufferIn[1024+i].r;

        in[i].r = (bufferIn[i].r * (0.5 - 0.5 * cos((2 * M_PI * i) / (2*FRAME_SIZE - 1))));
        in[i].i = 0;
        in[1024+i].r = (bufferIn[1024+i].r * (0.5 - 0.5 * cos((2 * M_PI * (1024+i)) / (2*FRAME_SIZE - 1))));
        in[1024+i].i = 0;
    }

    kiss_fft_cfg fft_cfg = kiss_fft_alloc(2*FRAME_SIZE, 0, NULL, NULL);
    kiss_fft(fft_cfg, in, out);
    free(fft_cfg);

    if(energy <= VOICED_THRESHOLD) {
        flag = 0;

        for(int i=0; i < 2*FRAME_SIZE; i++) {
            float mag = sqrt(out[i].r*out[i].r + out[i].i*out[i].i);
            if(abs(mag - noise_spec[i]) > residual[i]) {residual[i] = abs(mag - noise_spec[i]);}


            if(first_frame) {noise_spec[i] = sqrt(out[i].r*out[i].r + out[i].i*out[i].i);}
            else {noise_spec[i] = (noise_spec[i] + sqrt(out[i].r*out[i].r + out[i].i*out[i].i))/2;}

            //out[i].r *= 0.0001; out[i].i *= 0.0001;
        }
    }

    else {
        for(int i=0; i < 2*FRAME_SIZE; i++) {
            float unit_r = 0.0;
            float unit_i = 0.0;

            float mag = sqrt(out[i].r*out[i].r + out[i].i*out[i].i);
            if(mag > 0.0) {
                unit_r = out[i].r/mag;
                unit_i = out[i].i/mag;
            }

            if(mag > noise_spec[i]) {
                out[i].r = (mag - noise_spec[i])*unit_r;
                out[i].i = (mag - noise_spec[i])*unit_i;
            }
        }
    }

    first_frame = false;

    if(curr_flag > 0) {
    for(int i=0; i<=2*FRAME_SIZE; i++) {
        float unit_r = 0.0;
        float unit_i = 0.0;
        float mag_c = sqrt(curr[i].r*curr[i].r + curr[i].i*curr[i].i);
        if(mag_c > 0.0) {
            unit_r = curr[i].r/mag_c;
            unit_i = curr[i].i/mag_c;
        }

        transition = curr[i];

        if(mag_c < residual[i]) {
            float mag_p = sqrt(prev[i].r*prev[i].r + prev[i].i*prev[i].i);
            float mag_f = sqrt(out[i].r*out[i].r + out[i].i*out[i].i);

            mag_c = fmin(mag_c, mag_p);
            mag_c = fmin(mag_c, mag_f);

            curr[i].r = mag_c * unit_r;
            curr[i].i = mag_c * unit_i;
        }

        if(flag > 0) {prev[i] = transition;}
    }
  }

    else {
        for(int i=0; i<2*FRAME_SIZE; i++) {
            curr[i].r *= 0.0001; curr[i].i *= 0.0001;
        }
    }

    kiss_fft_cfg ifft_cfg = kiss_fft_alloc(2*FRAME_SIZE, 1, NULL, NULL);
    kiss_fft(ifft_cfg, curr, in);
    free(ifft_cfg);

    for(int i=0; i<2*FRAME_SIZE; i++) {
        in[i].r = in[i].r/2048.0;
    }

    for(int i=0; i < FRAME_SIZE; i++) {
        int16_t tmp = (output_buff[i] + ((int16_t) in[i].r));
        dataBuf->buf_[i*2] = tmp;
        dataBuf->buf_[i*2+1] = ((uint16_t) tmp)>>8;
        output_buff[i] = ((int16_t) in[i+1024].r);
    }

    for(int i=0; i<2*FRAME_SIZE; i++) {
        curr[i] = out[i];
    }

    curr_flag = flag;

    gettimeofday(&end, NULL);
    LOGD("Loop timer: %ld us",  ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
}

void ece420ProcessFrame_crowd(sample_buf *dataBuf) {
  // Keep in mind, we only have 20ms to process each buffer!
      struct timeval start;
      struct timeval end;
      gettimeofday(&start, NULL);

      long long energy=0;

      // Data is encoded in signed PCM-16, little-endian, mono
      for (int i = 0; i < FRAME_SIZE; i++) {
          int16_t val = ((uint16_t) dataBuf->buf_[2 * i]) | (((uint16_t) dataBuf->buf_[2 * i + 1]) << 8);

          bufferIn_w[i].r = ((float) val);
          bufferIn_w[i].i = 0;
          energy = energy + bufferIn_w[i].r*bufferIn_w[i].r;
      }

      kiss_fft_cfg fft_cfg = kiss_fft_alloc(FRAME_SIZE, 0, NULL, NULL);
      kiss_fft(fft_cfg, bufferIn_w, out_w);
      free(fft_cfg);

      if(energy <= VOICED_THRESHOLD) {    //unvoiced


          for(int i=0; i < FRAME_SIZE; i++) {
              if(first_frame_w) {noise_spec_w[i] = sqrt(out_w[i].r*out_w[i].r + out_w[i].i*out_w[i].i);}
              else {noise_spec_w[i] = (noise_spec_w[i] + sqrt(out_w[i].r*out_w[i].r + out_w[i].i*out_w[i].i))/2;}

              //attenuate output cuz it is noise
              out_w[i].r *= 0.0001; out_w[i].i *= 0.0001;
          }

      }

      else {      //voiced


          for(int i=0; i < FRAME_SIZE; i++) {
              float s = 0.0;
              float n = 0.0;

              float mag = sqrt(out_w[i].r*out_w[i].r + out_w[i].i*out_w[i].i);

              s =  (mag - noise_spec_w[i])*(mag - noise_spec_w[i]);
              n =  noise_spec_w[i]*noise_spec_w[i];

              if(s+n !=0){
                  wiener[i] = pow((s/(s+n)),2);
              }
              else{
                  wiener[i] = 0;
              }

              out_w[i].r = wiener[i] *out_w[i].r;
              out_w[i].i = wiener[i] * out_w[i].i;
          }
      }




      first_frame_w = false;

      //take ifft
      kiss_fft_cfg ifft_cfg = kiss_fft_alloc(FRAME_SIZE, 1, NULL, NULL);
      kiss_fft(ifft_cfg, out_w, in_w);
      free(ifft_cfg);


      //ifft divide by 1024
      for(int i=0; i<FRAME_SIZE; i++) {
          in_w[i].r = in_w[i].r/1024.0;
      }


      //convert back to buffer
      for(int i=0; i < FRAME_SIZE; i++) {
          int16_t tmp = ((int16_t) in_w[i].r);
          dataBuf->buf_[i*2] = tmp;
          dataBuf->buf_[i*2+1] = ((uint16_t) tmp)>>8;
      }

      gettimeofday(&end, NULL);
      LOGD("Loop timer: %ld us",  ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
}

void ece420ProcessFrame_helicopter(sample_buf *dataBuf) {
// Keep in mind, we only have 20ms to process each buffer!
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    gettimeofday(&end, NULL);
    LOGD("Loop timer: %ld us",  ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
}

void ece420ProcessFrame_engine(sample_buf *dataBuf) {
// Keep in mind, we only have 20ms to process each buffer!
    struct timeval start;
    struct timeval end;
    float D, noise_energy, speech_energy, SNR;
    float alpha, beta;
    gettimeofday(&start, NULL);

    long long energy=0;

    // Data is encoded in signed PCM-16, little-endian, mono
    for (int i = 0; i < FRAME_SIZE; i++) {
        int16_t val = ((uint16_t) dataBuf->buf_[2 * i]) | (((uint16_t) dataBuf->buf_[2 * i + 1]) << 8);

        bufferIn_l[i].r = ((float) val);
        bufferIn_l[i].i = 0;
        energy = energy + bufferIn_l[i].r*bufferIn_l[i].r;
    }

    kiss_fft_cfg fft_cfg = kiss_fft_alloc(FRAME_SIZE, 0, NULL, NULL);
    kiss_fft(fft_cfg, bufferIn_l, out_l);
    free(fft_cfg);

    if(energy <= VOICED_THRESHOLD) {    //unvoiced

        for(int i=0; i < FRAME_SIZE; i++) {
            if(first_frame_l) {noise_spec_l[i] = sqrt(out_l[i].r*out_l[i].r + out_l[i].i*out_l[i].i);}
            else {noise_spec_l[i] = (noise_spec_l[i] + sqrt(out_l[i].r*out_l[i].r + out_l[i].i*out_l[i].i))/2;}

            //attenuate output cuz it is noise
            out_l[i].r *= 0.0001; out_l[i].i *= 0.0001;
        }

    }

    else {      //voiced
        noise_energy = 0.0; speech_energy = 0.0;

        for(int i=0; i<FRAME_SIZE; i++) {
            noise_energy += noise_spec_l[i]*noise_spec_l[i];
            speech_energy += out_l[i].r*out_l[i].r + out_l[i].i*out_l[i].i;
        }

        SNR = 20*log10(speech_energy/noise_energy);

        if(SNR < -5) {alpha = 4.75;}
        else if(SNR>20) {alpha = 1;}
        else {alpha = 4 - 0.15*SNR;}

        if(SNR<0) {beta = 0.04;}
        else {beta = 0.01;}

        for(int i=0; i < FRAME_SIZE; i++) {
            float unit_r = 0.0;
            float unit_i = 0.0;

            float mag = sqrt(out_l[i].r*out_l[i].r + out_l[i].i*out_l[i].i);
            if(mag > 0.0) {
                unit_r = out_l[i].r/mag;
                unit_i = out_l[i].i/mag;
            }

            D = mag - alpha*noise_spec_l[i];
            if(D > beta*noise_spec_l[i]) {
                out_l[i].r = (D)*unit_r;
                out_l[i].i = (D)*unit_i;
            }
            else {
                out_l[i].r = (beta*noise_spec_l[i])*unit_r;
                out_l[i].i = (beta*noise_spec_l[i])*unit_i;
            }
        }
    }


    first_frame_l = false;

    //take ifft
    kiss_fft_cfg ifft_cfg = kiss_fft_alloc(FRAME_SIZE, 1, NULL, NULL);
    kiss_fft(ifft_cfg, out_l, in_l);
    free(ifft_cfg);

    //ifft divide by 1024
    for(int i=0; i<FRAME_SIZE; i++) {
        in_l[i].r = in_l[i].r/1024.0;
    }

    //convert back to buffer
    for(int i=0; i < FRAME_SIZE; i++) {
        int16_t tmp = ((int16_t) in_l[i].r);
        dataBuf->buf_[i*2] = tmp;
        dataBuf->buf_[i*2+1] = ((uint16_t) tmp)>>8;
    }

    gettimeofday(&end, NULL);
    LOGD("Loop timer: %ld us",  ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
}
