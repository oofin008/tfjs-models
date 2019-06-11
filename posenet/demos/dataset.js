//Convert pose to L2 Norm > done
//Next1: create VP tree with L2 data > done
//Next2: create imageCollection > done
//Next3: match VP tree with imageCollection > done
//Next4: show result: video.onloadmetadata error, use localstorage > not enough memory
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';

const similarity = require('compute-cosine-similarity');
const tf = require('@tensorflow/tfjs');
const VPTreeFactory = require('vptree');
import {drawBoundingBox, drawKeypoints, drawSkeleton} from './demo_util';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

var imageScaleFactor = 0.5;
var outputStride = 16;
var flipHorizontal = false;

var _CANVAS = document.querySelector("#video-canvas"),
	_CTX = _CANVAS.getContext("2d"),
	_VIDEO = document.querySelector("#main-video");

var vid_duration_array = new Array();

var poseData = [];
var image_list = [];

//try{
//	localStorage.removeItem("Local_poseData");
//	localStorage.removeItem("Local_imageList");
//	console.log("remove local storage item success");
//}catch(e){
//	console.log("no item to delete: ", e);
//}

!function(){function e(t,o){return n?void(n.transaction("s").objectStore("s").get(t).onsuccess=function(e){var t=e.target.result&&e.target.result.v||null;o(t)}):void setTimeout(function(){e(t,o)},100)}var t=window.indexedDB||window.mozIndexedDB||window.webkitIndexedDB||window.msIndexedDB;if(!t)return void console.error("indexDB not supported");var n,o={k:"",v:""},r=t.open("d2",1);r.onsuccess=function(e){n=this.result},r.onerror=function(e){console.error("indexedDB request error"),console.log(e)},r.onupgradeneeded=function(e){n=null;var t=e.target.result.createObjectStore("s",{keyPath:"k"});t.transaction.oncomplete=function(e){n=e.target.db}},window.ldb={get:e,set:function(e,t){o.k=e,o.v=t,n.transaction("s","readwrite").objectStore("s").put(o)}}}();
//function for real time camera input
function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

//set delay when change frame
async function delay_change_frame(frame){
	_VIDEO.currentTime = frame;
	console.log('frame# :', frame);
	return new Promise(resolve => {
		_CTX.drawImage(_VIDEO, 0, 0, _VIDEO.videoWidth, _VIDEO.videoHeight);
		setTimeout(() => {
			resolve('success');
		}, 3000);
	});
}

let vptree;
function buildVPTree() {
  // Initialize our vptree with our images’ pose data and a distance function
  vptree = VPTreeFactory.build(poseData, cosineDistanceMatching);
  console.log('build VP tree success');
}
function cosineDistanceMatching(poseVector1, poseVector2){
	let cosineSimilarity = similarity(poseVector1, poseVector2);
	let distance = 2 * (1 - cosineSimilarity);
	return Math.sqrt(distance);
}
function findMostSimilarMatch(userPose) {
	// search the vp tree for the image pose that is nearest (in cosine distance) to userPose
	let nearestImage = vptree.search(userPose);
	if(nearestImage[0] == undefined){
		alert("undefine yea!!");
		return 0;
	}
	else{
		console.log(nearestImage);
		//console.log(nearestImage[0].d); // cosine distance value of the nearest match
		// return index (in relation to poseData) of nearest match. 
		return nearestImage[0].i;
	}
}

//showing result on website
let images;
function ImageCollection(list, callback){
	var total = 0, images = {};
	for(var i=0; i< list.length; i++){
		var img = new Image();
		images[list[i].name] = img;
		img.onload = function(){
			total++;
			if(total == list.length){
				callback && callback();
			}
		};
		img.src = list[i].url;
	}
	
	this.get = function(name){
		return images[name];
	};
}

//Image Object constructor function
function _Person(name, url){
	this.name = name;
	this.url = url;
}
//create list and push new person into list

//Upload video part
//Work better when set time to 2 second
async function call_change_frame(_Array){
	console.log('input array:', _Array);
	for(const [i, item] of _Array.entries()){
		
		await delay_change_frame(item)
		console.log('item: ', i);
		var person = new _Person(i, _CANVAS.toDataURL());
		image_list.push(person);
		posenet.load().then(function(net){
			return net.estimateSinglePose(_CANVAS, imageScaleFactor, flipHorizontal, outputStride)
		}).then(function(pose){
			const boundingBox = posenet.getBoundingBox(pose.keypoints);
			var test_distance = 0;
			var test_vec = new Array();
			for(var i=0; i<17; i++){
				var test_x = pose.keypoints[i].position.x -boundingBox.minX;
				var test_y = pose.keypoints[i].position.y - boundingBox.minY;
				test_vec.push(test_x);
				test_vec.push(test_y);
			}
			var norm=0;
			var test_L2 = new Array();

			for(var i=0; i<test_vec.length; i+=2){
				norm += Math.pow(test_vec[i], 2);
				norm += Math.pow(test_vec[i+1], 2);
				norm = Math.sqrt(norm);
				test_L2.push(test_vec[i]/norm);
				test_L2.push(test_vec[i+1]/norm);
				norm = 0;
			}
			poseData.push(test_L2);
		})
	}
	console.log('Function call_change_frame done');
	console.log('poseData length: ', poseData.length);
	console.log('image list length: ', image_list.length);
	
	//build local storage to send poseData and image_list
//	localStorage.setItem("Local_imageList", JSON.stringify(image_list));
//	localStorage.setItem("Local_poseData", JSON.stringify(poseData));
//	ldb.set("Local_poseData", JSON.stringify(poseData));
//	ldb.set("Local_imageList", JSON.stringify(image_list));

	
	console.log("build local poseData done");
	console.log("build local imageList done");
	console.log("poseData: ", poseData);
	console.log("imageList: ", image_list);
	
	buildVPTree();
	images = new ImageCollection(image_list, function(){
		console.log('Image collection loaded');
	});
//	_VIDEO.pause();
//	_VIDEO.removeAttribute('src');
//	_VIDEO.load();
}

document.querySelector("#start_stream_button").addEventListener('click', function(){
	navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
	bindPage();
});

document.querySelector("#process-button").addEventListener('click', function(){
	call_change_frame(vid_duration_array);
});

// Upon click this should should trigger click on the #file-to-upload file input element
// This is better than showing the not-good-looking file input element
document.querySelector("#upload-button").addEventListener('click', function() {
	document.querySelector("#file-to-upload").click();
});

// When user chooses a MP4 file
document.querySelector("#file-to-upload").addEventListener('change', function() {
	// Validate whether MP4
    if(['video/mp4'].indexOf(document.querySelector("#file-to-upload").files[0].type) == -1) {
        alert('Error : Only MP4 format allowed');
        return;
    }

    // Hide upload button
    document.querySelector("#upload-button").style.display = 'none';

	// Object Url as the video source
	document.querySelector("#main-video source").setAttribute('src', URL.createObjectURL(document.querySelector("#file-to-upload").files[0]));
	
	// Load the video and show it
	_VIDEO.load();
	_VIDEO.style.display = 'inline';
	
	// Load metadata of the video to get video duration and dimensions
	_VIDEO.addEventListener('loadedmetadata', function() {
		
	    var video_duration = _VIDEO.duration,
	    	duration_options_html = '';
		
		console.log("video duration : ", video_duration);
	    // Set options in dropdown at 4 second interval
	    for(var i=0; i<Math.floor(video_duration); i=i+4) {
	    	duration_options_html += '<option value="' + i + '">' + i + '</option>';
			vid_duration_array.push(i);
	    }
	    document.querySelector("#set-video-seconds").innerHTML = duration_options_html;
	    
	    // Show the dropdown container
	    document.querySelector("#thumbnail-container").style.display = 'block';

	    // Set canvas dimensions same as video dimensions
	    _CANVAS.width = _VIDEO.videoWidth;
		_CANVAS.height = _VIDEO.videoHeight;
	});
});

// On changing the duration dropdown, seek the video to that duration
document.querySelector("#set-video-seconds").addEventListener('change', function() {
    _VIDEO.currentTime = document.querySelector("#set-video-seconds").value;
    
    // Seeking might take a few milliseconds, so disable the dropdown and hide download link 
    document.querySelector("#set-video-seconds").disabled = true;
    document.querySelector("#get-thumbnail").style.display = 'none';
});

// Seeking video to the specified duration is complete 
document.querySelector("#main-video").addEventListener('timeupdate', function() {
	// Re-enable the dropdown and show the Download link
	_CTX.drawImage(_VIDEO, 0, 0, _VIDEO.videoWidth, _VIDEO.videoHeight);
	document.querySelector("#set-video-seconds").disabled = false;
    document.querySelector("#get-thumbnail").style.display = 'inline';
});

// On clicking the Download button set the video in the canvas and download the base-64 encoded image data
document.querySelector("#get-thumbnail").addEventListener('click', function() {
    _CTX.drawImage(_VIDEO, 0, 0, _VIDEO.videoWidth, _VIDEO.videoHeight);

	document.querySelector("#get-thumbnail").setAttribute('href', _CANVAS.toDataURL());
	console.log('canvas url: ', typeof _CANVAS.toDataURL(), _CANVAS.toDataURL());
	console.log('_VIDEO: ', _VIDEO);
	const result_canvas = document.getElementById('result_img');
	const result_ctx = result_canvas.getContext('2d');
	result_canvas.width = videoWidth;
	result_canvas.height = videoHeight;
	var myImage = new Image();
	myImage.src = _CANVAS.toDataURL();
	myImage.onload = function(){
		result_ctx.drawImage(myImage, 0, 0, 600, 600);
	};
	document.querySelector("#get-thumbnail").setAttribute('download', 'thumbnail.jpg');
});

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
	console.log('stream: ', stream);
  video.srcObject = stream;
  return video;
//  return new Promise((resolve) => {
//    video.onloadedmetadata = () => {
//		console.log("stream load success: ");
//      resolve(video);
//    };
//  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const guiState = {
  algorithm: 'multi-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = input.add(
      guiState.input, 'mobileNetArchitecture',
      ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
      .min(1)
      .max(20)
      .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();


  architectureController.onChange(function(architecture) {
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    }
  });
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
	const canvas = document.getElementById('output');
	const ctx = canvas.getContext('2d');
	// since images are being fed from a webcam
	const flipHorizontal = true;

	canvas.width = videoWidth;
	canvas.height = videoHeight;
	
	//canvas for result img
	const result_canvas = document.getElementById('result_img');
	const result_ctx = result_canvas.getContext('2d');
	result_canvas.width = videoWidth;
	result_canvas.height = videoHeight;

	async function poseDetectionFrame() {
		if (guiState.changeToArchitecture) {
	  		// Important to purge variables and free up GPU memory
			guiState.net.dispose();

	  		// Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
	  		// version
	  		guiState.net = await posenet.load(+guiState.changeToArchitecture);

	  		guiState.changeToArchitecture = null;
		}

		// Begin monitoring code for frames per second
		stats.begin();

		// Scale an image down to a certain factor. Too large of an image will slow
		// down the GPU
		const imageScaleFactor = guiState.input.imageScaleFactor;
		const outputStride = +guiState.input.outputStride;

		let poses = [];
		let minPoseConfidence;
		let minPartConfidence;
		switch (guiState.algorithm) {
	  		case 'single-pose':
				const pose = await guiState.net.estimateSinglePose(
					video, imageScaleFactor, flipHorizontal, outputStride);
				poses.push(pose);

			minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
			minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
			break;
	  		case 'multi-pose':
				poses = await guiState.net.estimateMultiplePoses(
					video, imageScaleFactor, flipHorizontal, outputStride,
				guiState.multiPoseDetection.maxPoseDetections,
				guiState.multiPoseDetection.minPartConfidence,
				guiState.multiPoseDetection.nmsRadius);

				minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
				minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
				break;
		}

		ctx.clearRect(0, 0, videoWidth, videoHeight);

		if (guiState.output.showVideo) {
			ctx.save();
			ctx.scale(-1, 1);
		  	ctx.translate(-videoWidth, 0);
		  	ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
		  	ctx.restore();
		}

		// For each pose (i.e. person) detected in an image, loop through the poses
		// and draw the resulting skeleton and keypoints if over certain confidence
		// scores
		poses.forEach(({score, keypoints}) => {
		  if (score >= minPoseConfidence) {
			if (guiState.output.showPoints) {
			  drawKeypoints(keypoints, minPartConfidence, ctx);
			}
			if (guiState.output.showSkeleton) {
			  drawSkeleton(keypoints, minPartConfidence, ctx);
			}
			if (guiState.output.showBoundingBox) {
			  drawBoundingBox(keypoints, ctx);
			}


			//<-----get bounding box for rescaling
			const boundingBox = posenet.getBoundingBox(keypoints);
			var test_distance = 0;
			console.log("minX: " + boundingBox.minX);  
			console.log("minY: " + boundingBox.minY); 

			//Rescale
			var pose_rescale = new Array();
			for(var i=0; i<17; i++){
				var rescale_x = poses[0].keypoints[i].position.x - boundingBox.minX;
				var rescale_y = poses[0].keypoints[i].position.y - boundingBox.minY;
				pose_rescale.push(rescale_x);
				pose_rescale.push(rescale_y);
			}//Array length = 34 for "pose_rescale" array


			//<-----L2 Norm
			var norm=0;
			var pose_L2 = new Array();

			for(var i=0; i<pose_rescale.length; i+=2){
				norm += Math.pow(pose_rescale[i], 2);
				norm += Math.pow(pose_rescale[i+1], 2);
				norm = Math.sqrt(norm);
				pose_L2.push(pose_rescale[i]/norm);
				pose_L2.push(pose_rescale[i+1]/norm);
				norm = 0;
			}// pose_L2 length = 34

			let show_key = JSON.stringify(poses);
			document.getElementById('skeletonValue').innerHTML = show_key;
			let closestMatchIndex = findMostSimilarMatch(pose_L2);
			let closestMatch = poseData[closestMatchIndex];
//			var myImage = new Image();
//			myImage.src = images.get(closestMatchIndex);
//			myImage.onload = function(){
//				result_ctx.drawImage(myImage, 0, 0, 600, 600);
//			};
			result_ctx.drawImage(images.get(closestMatchIndex), 0, 0, 600, 600);
		  }
		});

	// End monitoring code for frames per second
	stats.end();
	requestAnimationFrame(poseDetectionFrame);
	}
	poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);
	console.log('posenet loaded');
  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'inline-block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  setupFPS();
  buildVPTree();
  detectPoseInRealTime(video, net);
}
//navigator.getUserMedia = navigator.getUserMedia ||
//    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
//// kick off the demo
//bindPage();
