<template>
  <div class="column">
    <h2>PopIn - upload a picture of your puppy or kitten to find out how popular it will be on Instagram!</h2>
    <div class="drop" 
      :class="getClasses" 
      @dragover.prevent="dragOver" 
      @dragleave.prevent="dragLeave"
      @drop.prevent="drop($event)">
      <div class="column">
        <img :src="imageSource" v-if="imageSource" />
        <h1 v-if="wrongFile">Wrong file type</h1>
        <h1 v-if="!imageSource && !isDragging && !wrongFile">Drop an image</h1>
        <h1 v-if="imageSource && !ranking">LOADING</h1>
        <h1 v-if="imageSource && ranking">{{score}}/10!</h1>
      </div>
      <div v-if="imageSource && ranking" id="progress-bar-container">
        <div class="progress-bar-child progress"></div>
        <div class="progress-bar-child shrinker timelapse" :style="cssVars" ></div>
      </div>
      <label class="manual" for="uploadmyfile">
          <p>or pick from device</p>
          <input type="file" id="uploadmyfile" :accept="'image/*'" @change="requestUploadFile">
      </label>
    </div>
  </div>
</template>


<script>
import axios from 'axios';
export default {
  name: 'DropAnImage',
  data(){
    return{
      isDragging:false,
      wrongFile:false,
      imageSource:null,
      ranking:null,
      score:null,
      width:60
    }
  },
  computed:{
    getClasses(){
      return {isDragging: this.isDragging}
    },
    cssVars() {
      return {
        '--width': this.width + '%'
      }
    }
  },
  methods:{
    dragOver(){
      this.isDragging = true
    },
    dragLeave(){
      this.isDragging = false
    },
    drop(e){
      let files = e.dataTransfer.files

      this.wrongFile = false

      if (files.length === 1) {

        let file = files[0]
        
        if (file.type.indexOf('image/') >= 0) {

          var reader = new FileReader()
          reader.onload = f => {
            this.imageSource = f.target.result
            this.isDragging = false
          }
          reader.readAsDataURL(file)
          this.sendToLambda(file)
        }else{
          this.wrongFile = true
          this.imageSource = null
          this.isDragging = false
        }
      }
    },
    requestUploadFile(){
      var src = this.$el.querySelector('#uploadmyfile')
      this.drop({dataTransfer:src})
    },

    getBase64(file) {
        const reader = new FileReader()
        return new Promise(resolve => {
            reader.onload = ev => {
                resolve(ev.target.result)
            }
            reader.readAsDataURL(file)
        })
    },

    sendToLambda(file) {
      this.ranking = null;
      this.getBase64(file).then(data => {
        return axios.post("https://d3jmnpsj1wjh47.cloudfront.net/api", {
          "body": data
        })}).then((response) => {
          this.ranking = Math.min((response.data / 0.3), 1)
          this.score = Math.round(this.ranking*10)
          this.width = 100-Math.round(this.ranking*100)
        }, (error) => {
          this.ranking = Math.min((0.1 / 0.3), 1)
          this.score = Math.round(this.ranking*10)
          this.width = 100-Math.round(this.ranking*100)
          console.log(error)
        });
    }
  }
}
</script>



<style scoped>
.drop{
  aspect-ratio: 1/2;
  height: 60vh;
  max-width: 360px;
  max-height: 720px;
  background-color: #eee;
  border:10px solid #eee;
  border-radius: 40px;

  display: flex;
  align-items: center;
  justify-content: center;

  padding: 1rem;
  margin-top: 5vh;

  font-family: sans-serif;
  
  overflow: hidden;
  position: relative;
  box-shadow: 0px 0px 0px 11px #1f1f1f, 0px 0px 0px 13px #191919, 0px 0px 0px 20px #111;
  left: 5%;
  top: 5%;
}

.isDragging{
  background-color: #999;
  border-color: #fff;
}

img{
  width: 100%;
  height: 100%;
  max-height: 25vh;
  object-fit: contain;
}

.manual{
  position: absolute;
  bottom:0;
  width:100%;
  text-align:center;
  font-size:.8rem;
  text-decoration: underline;
}
#uploadmyfile{
  display: none;
}

.manual:hover {
  background-color:  #999;
  cursor: pointer;
}

#progress-bar-container {
	width: 90%;
	height: 6%;
  bottom:15%;
	position: absolute;
	border-radius: 40px;
	overflow: hidden;
}

.progress-bar-child {
	width: 100%;
	height: 100%;
}

.progress {
	color: white;
	text-align: center;
	line-height: 75px;
	font-size: 35px;
	font-family: "Segoe UI";
	animation-direction: reverse;
	background: #e5405e;
	background: linear-gradient(to right, #e04f6a 0%, #ffdb3a 45%, #3fffa2 100%);
}

.shrinker {
	background-color: black;
	position: absolute;
	top: 0;
	right: 0;
  width: var(--width);
}

.timelapse {
	animation-name: timelapse;
	animation-fill-mode: forwards;
	animation-duration: 2s;
	animation-timing-function: cubic-bezier(.86, .05, .4, .96);
}

@keyframes timelapse {
	0% {
		width: 100%;
	}
	100% {
		width: var(--width);
	}
}

</style>
