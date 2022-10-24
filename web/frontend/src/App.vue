<template>
  <div id="app">

    <div class="parent-wrapper">
      <DropAnImage />
    </div>
    
  </div>
</template>

<script>
import DropAnImage from './components/DropAnImage.vue'
import TriggerUploadDialog from './components/TriggerUploadDialog.vue'

export default {
  name: 'app',
  components: {
    DropAnImage, TriggerUploadDialog
  }
}
</script>


<style>
*{
  box-sizing: border-box;
}

html, body, #app{
  width:100%;
  height:100%;
  padding: 0;
  margin: 0;
  overflow: hidden;
}
body{
  background-color: #fff;
  padding:1rem;
}

.parent-wrapper{
  margin-bottom:1rem;
  width:100%;
  height:100%; /* was 50% */
}
</style>