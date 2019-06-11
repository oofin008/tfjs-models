/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
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

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

//Function for compare pose and finding matching, vp tree algorithm
const poseData = [  [0.9926248076968573,0.12122702316223469,0.9999312298509863,0.011727555955717733,1,0,0.9963692856287516,0.08513663521452654,0.9974931474308112,0.07076313184543202,0.8483313003958096,0.5294657729908089,0.45484701410539,0.8905695895096638,0.7499132169083468,0.6615362175317197,0.01778016532136929,0.9998419203659872,0.9423461551442,0.3346396926321554,0,1,0.38774470708198683,0.9217668046365655,0.16102280265914137,0.9869506862167913,0.21127378748507203,0.9774269214226261,0.0496737364980564,0.9987654979535094,0.21419996735965588,0.9767898310195097,0.19527157979311066,0.9807492085773524],
					[0.995090545740632,0.09896871110412248,0.9985977923367536,0.05293816336219189,0.9999868751785946,0.0051234237136683755,0.9969205910057605,0.07841769716540588,1,0,0.8896734382508502,0.45659738640394176,0.5410296046469382,0.8410035474928615,0.6655917335568997,0.746316048481286,0.10816789289774857,0.9941326405193932,0.5935729096138126,0.8047802190490224,0.5855518954533824,0.8106349225952155,0.4401968830528588,0.8979012775080275,0.188654896320412,0.9820434461337922,0.3897662779183983,0.9209138117096723,0.05479322117847572,0.9984977230384087,0.33846063037518137,0.9409805532985445,0,1],
					[0.992350742424636,0.12345041113448682,1,0,0.9999851565671891,0.005448545245707304,0.9967431846354323,0.08064132862754715,0.9888521916371441,0.1489004469248382,0.8389923853542726,0.5441431588447544,0.3484596920709528,0.9373237663698796,0.6645584024038111,0.7472363279408291,0,1,0.4414025382129615,0.897309199361711,0.09722515794762142,0.9952624119608156,0.415991453526337,0.9093685229834191,0.1551642708888931,0.9878886825141374,0.27832010018286735,0.9604883767303999,0.09156916851028907,0.9957987183052277,0.21227179387588496,0.9772106658877162,0.0719275181220756,0.9974098616600894],
					[0.6400188019849736,0.7683592474264361,0.9925732510923685,0.1216484328543779,1,0,0.98684829934416,0.1616491078897108,0,1,0.8628314114573619,0.5054917955837628,0.48942493175167096,0.8720454324058305,0.5882082384971912,0.8087095078976329,0.10540964326894962,0.9944288848911785,0.5458314561959927,0.8378949942725293,0.23888141400830143,0.9710487475103372,0.3961213580242833,0.9181981647318826,0.3680595779861258,0.9298022085651738,0.13112250493538005,0.9913661728642306,0.14227967536665298,0.98982649690618,0.16356931512068387,0.9865318439619424,0.15793893003306114,0.9874488819073176],
					[0.9914146375870901,0.13075555964493038,0.9995053681754533,0.03144867228439931,0.9997072727703414,0.024194395427168044,0.9980077740337129,0.0630910688471313,1,0,0.8869208746817518,0.46192138081469697,0.5237705893423474,0.8518593603054263,0.7697711627652752,0.6383199487521881,0,1,0.7413375872939725,0.6711323130837553,0.19896656452218267,0.9800062786544994,0.5092164222866041,0.8606385044103192,0.29933473423555773,0.9541481629600971,0.522471888760305,0.8526565108267452,0.2442355988122751,0.9697159234914158,0.26231021297034635,0.9649836020220507,0.24552158672524718,0.9693911235677356],
					[0.8391134067106489,0.5439565154297255,1,0,0.992107417724529,0.12539087564877618,0.9576385035030734,0.2879730831317293,0.5489334180963061,0.8358660792837007,0.4528031764678216,0.8916104998151664,0.21746913709755858,0.9760671976918613,0.2256854897310763,0.974200215420241,0.04967775818398471,0.9987652979263013,0.1387551496017627,0.9903267180375336,0.08865295131206118,0.9960625754558101,0.081667934510919,0.9966595950838583,0.0921554790840215,0.9957446297494125,0.0007192824628728927,0.9999997413163358,0.08347509046352085,0.9965098641117944,0.04836660605300466,0.9988296508508914,0,1],
					[0.9655479167478799,0.26022532633052725,1,0,0.9991653169297886,0.04084935060188133,0.9751853424195807,0.221390035751398,0.9935055316099137,0.11378382424801413,0.8028321073754944,0.5962051722075402,0.5984651189301414,0.8011488634604255,0.7001870936994354,0.713959406280734,0.607188039409119,0.7945581695499142,0.00897274848245633,0.9999597440820659,0,1,0.35889048462197565,0.9333796762560258,0.37222040506578963,0.9281443691865288,0.5486067835991681,0.8360804967160611,0.5208058196230907,0.8536751713893996,0.05320652645791333,0.9985835295769119,0.07634794173440146,0.9970812363057087],
					[0.9931581492534406,0.11677709780380977,0.999644713077294,0.02665422324163286,1,0,0.9864929845343295,0.16380351481134564,0.9915218396736553,0.12994014564471681,0.8891502355558025,0.45761540469160433,0.5827727960529308,0.8126351384112364,0.7390981493801421,0.6735977476082065,0,1,0.679410401835936,0.7337584792539925,0.17128058336639865,0.985222290532277,0.47466780740194103,0.8801650257856386,0.1924274174050128,0.9813112090620576,0.4000166779525908,0.9165078599552615,0.1934321818472399,0.9811136483740384,0.39014315680680384,0.92075421106657,0.18023522910138917,0.9836235368222792],
					[0.9753402845429303,0.22070643250189997,0.9999963840021474,0.002689234580651671,0.9976142887147474,0.06903427374984238,0,1,1,0,0.827860915392696,0.5609334227563619,0.7566190041420344,0.6538560105796353,0.9558041966136465,0.2940039757141757,0.9597305209758406,0.2809222794785804,0.8134544291400333,0.5816286544802125,0.29542392175877696,0.9553662682200288,0.43772457387208275,0.8991091131940014,0.5365921068351069,0.8438417570150587,0.36727603992179414,0.9301119881494726,0.2909640509513817,0.9567339865678243,0.22282318553691574,0.9748588759339382,0.23597146896527946,0.9717599836556186],
					[0.9985120031377549,0.05453237194390279,0.9999988098186793,0.001542841931283194,1,0,0.9996413314474536,0.02678074797242062,0.9990027916159786,0.044647758549358006,0.9575144010337907,0.2883854570065921,0.8918803511655522,0.4522714220518595,0.8844817821714306,0.46657472821280155,0.6049303937093432,0.7962783550785862,0.8702644498389613,0.49258480218789724,0,1,0.6650537395880229,0.7467955031064304,0.556943245085499,0.8305505534003538,0.4538649243596254,0.8910704968946237,0.44574846694580816,0.8951582565197406,0.35469097337779876,0.9349835899117748,0.3589352575817313,0.9333624595325956],
					[0.9822213208672788,0.18772660129491023,0.9999702311698896,0.007716007648885064,1,0,0.9962891278369754,0.08606958669494663,0.9736506907435927,0.2280445842692276,0.8772688459602445,0.47999934573661746,0.30445028082746556,0.952528228717699,0.619284188699003,0.7851669208693256,0,1,0.42095131910970185,0.9070832304368778,0.4689821972369399,0.8832076192350312,0.3886251329447365,0.9213959550832019,0.16814443530354836,0.985762369375323,0.24768459967351977,0.9688407191507633,0.14919455312554908,0.9888078606674138,0.17438440203623726,0.9846776530045068,0.12130456359448578,0.9926153347854099],
					[0.9775216248533163,0.21083517956956957,1,0,0.9988230051048479,0.048503654226471564,0.9983208882998911,0.057925849015068476,0.972836749753855,0.23149224247986197,0.8733670163468111,0.4870626805221979,0.30980841994468084,0.950799002382407,0.5908838755612009,0.806756621046134,0,1,0.266763412568761,0.9637620462098874,0.31366155745463364,0.949534847899293,0.38099977844980776,0.9245751288138769,0.12833948605515869,0.9917302941420604,0.2704190104989416,0.9627427271918356,0.0883370502893159,0.9960906412300954,0.22025094492953354,0.975443243483519,0.10037884817774409,0.994949288576312],
					[0.9852989059565738,0.1708392985257751,0.9999519773039408,0.009800157444600843,0.998772109294664,0.04954062671270514,1,0,0.9981049773758289,0.06153417048759057,0.9422974099896797,0.33477692741098686,0.642631222718323,0.7661756401684624,0.8836652102290145,0.46811942518006194,0,1,0.8870648828328762,0.46164476997437714,0.30052105453443156,0.953775180942298,0.5217263516829553,0.8531128963739754,0.34246640248504534,0.9395300757128273,0.405793174847348,0.9139649332700405,0.18328601611842046,0.9830596300812267,0.32789871876470916,0.944712882431727,0.18605685207215733,0.9825389802939114],
					[0.9979693712985088,0.06369563525124002,0.9993558986642345,0.03588576047682944,0.9990030443551914,0.044642103098527515,0.9995726077617745,0.029233573375931615,1,0,0.8780094789621722,0.4786432438179556,0.9378634425961063,0.34700455766715843,0.5782416005426851,0.815865584150866,0.7281125279695586,0.6854576183935654,0,1,0.603319129828175,0.7974998605538269,0.5439133961098457,0.8391413573005767,0.6328668816420434,0.7742606215743352,0.3631505120066152,0.9317304898034267,0.4921476751081889,0.8705117264492217,0.41489715010759753,0.9098683173034401,0.44077631247102794,0.8976169797661153],
					[0.9966257957243215,0.0820793719326784,1,0,0.99999040097818,0.0043815467016808714,0.9972231177405347,0.07447182986772725,0.995109905780651,0.09877385998949263,0.9510215854649102,0.3091244797485451,0.8630635610960018,0.5050953271495273,0.8636306214019435,0.5041251330541784,0.32271561798780796,0.9464959745856013,0.7970304135324269,0.6039391690429828,0,1,0.771182585513835,0.6366140273353994,0.6241700719249004,0.7812885006918153,0.8371084696673075,0.5470369366059835,0.5509320577994262,0.8345500989688335,0.8682321012978654,0.4961582593043202,0.542590069817869,0.8399976286484624],
					[0.9882993007201722,0.15252702119958456,0.9994452656555787,0.033304068199691214,0.9975068826964899,0.07056924948680778,1,0,0.9837867820461965,0.17934204044559496,0.6476804973191447,0.7619120509562934,0.571206167324344,0.820806624248753,0.29258591758134844,0.9562392382835376,0.20892178371388848,0.9779323536369001,0.37611324883577213,0.9265737013590449,0.31712703840386336,0.9483830668633823,0.1271456905451099,0.9918840523850595,0.08328062164465361,0.9965261351607794,0.08388880305685939,0.9964751219783098,0.08450557604645721,0.9964230063668023,0.006117638217534434,0.9999812870762329,0,1],
					[0.6248814329359351,0.7807196646504636,1,0,0,1,0.9657892081207139,0.2593283738381983,0.9968492786295214,0.07931907523290205,0.5819576616647287,0.813219084890241,0.1465899836526019,0.9891973396106207,0.5294039924758903,0.8483698561067499,0.08281440112266465,0.9965649878290399,0.1623692525280283,0.9867300673606177,0.11133629030203322,0.9937827883706687,0.39715699857083575,0.9177506842744414,0.24479797418520413,0.9695741084800172,0.18911984575331536,0.9819540131504337,0.005328164516222546,0.9999858052306985,0.22427649816720807,0.9745255524458322,0.20485570921850807,0.9787921834590744],
					[0.9568668740889532,0.29052673761847686,0.9895173421150312,0.14441409091083904,1,0,0.9975435443680141,0.07004910484581442,0.9903237574983648,0.1387762779956279,0.7593150897405166,0.6507231319788402,0.45755654869344653,0.8891805242737505,0.6130436837006293,0.7900490123244018,0,1,0.3106301394330048,0.9505308603490115,0.18835153281288797,0.9821016750250635,0.40603768152043357,0.9138563350907575,0.25924853780404394,0.9658106417132011,0.2438539416818012,0.9698119689539043,0.16617352963457133,0.9860965257259494,0.16251466514872556,0.986706128293322,0.16795503433780917,0.9857946573402523],
					[0,1,1,0,0.9997282624599128,0.023310968209918407,0.9893847889845818,0.1453194389128113,0.9763429652980886,0.21622769043981213,0.9725472508949179,0.23270548935669247,0.2630212715755974,0.9647900345146377,0.8052500740792545,0.5929353406530558,0.46765157265327306,0.883912895366348,0.5282904101511401,0.8490637446872527,0.3317848706765036,0.9433550760928654,0.32078775994741465,0.9471511036090914,0.17770320407802256,0.9840841281416974,0.17902929917503593,0.9838437426933687,0.1906830876314588,0.9816516490544562,0.1932582537850857,0.9811479232735191,0.2060629900353012,0.978538728992221],
					[0.949601047486824,0.31346108308995346,1,0,0.9971459158388907,0.07549849353344813,0.9890684951734592,0.14745681352622836,0.9424330613125356,0.33439486381384864,0.6418848734510366,0.7668010232353284,0.32467927430874033,0.9458241743761627,0.4887734865486066,0.872410728296666,0,1,0.4276849176623666,0.9039278794263039,0.041982735000838184,0.9991183363154984,0.16442397715697507,0.9863897585315261,0.10073863265978857,0.9949129247777597,0.1486110133703606,0.988895730957028,0.07735831078369675,0.9970033559385308,0.06039201068174387,0.9981747367299054,0.01689461644004409,0.9998572757825708],
					[0.9927080938721377,0.12054310582006336,1,0,0.9983557708944604,0.057321503135628844,0.9986010804122728,0.05287610234729396,0.9965602692418742,0.0828711636732781,0.8913685135849753,0.45327935425001625,0.646465611438983,0.7629431258139902,0.8913742354419006,0.4532681021099622,0.11981510684819395,0.9927962228831029,0.7969871200361458,0.6039963000685437,0,1,0.4918931679079667,0.8706555641385776,0.3255169680492825,0.9455361989432253,0.31987349875156257,0.9474602602729225,0.17195425488380767,0.9851049356425712,0.2226222702815182,0.974904777285814,0.15422995824992158,0.9880349791268664]
				 ];
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


//draw image on canvas
//function MydrawImage(img){
//	const Mycanvas = document.getElementById('result_img');
//  	const Myctx = Mycanvas.getContext('2d');
//	ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
//	ctx.drawImage(images.get(img), 0, 0, 600, 600);
//}

//function load image from Movemirror


var images = new ImageCollection([{
	name: "0",
	url: "/images/0001.jpg"
},{
	name: "1",
	url: "/images/0002.jpg"
},{
	name: "2",
	url: "/images/0003.jpg"
},{
	name: "3",
	url: "/images/0004.jpg"
},{
	name: "4",
	url: "/images/0005.jpg"
},{
	name: "5",
	url: "/images/0006.jpg"
},{
	name: "6",
	url: "/images/0007.jpg"
},{
	name: "7",
	url: "/images/0008.jpg"
},{
	name: "8",
	url: "/images/0009.jpg"
},{
	name: "9",
	url: "/images/0010.jpg"
},{
	name: "10",
	url: "/images/0011.jpg"
},{
	name: "11",
	url: "/images/0012.jpg"
},{
	name: "12",
	url: "/images/0013.jpg"
},{
	name: "13",
	url: "/images/0014.jpg"
},{
	name: "14",
	url: "/images/0015.jpg"
},{
	name: "15",
	url: "/images/0016.jpg"
},{
	name: "16",
	url: "/images/0017.jpg"
},{
	name: "17",
	url: "/images/0018.jpg"
},{
	name: "18",
	url: "/images/0019.jpg"
},{
	name: "19",
	url: "/images/0020.jpg"
},{
	name: "20",
	url: "/images/0021.jpg"
}], function(){
	console.log("Image Collection loaded");
});

console.log("Image collection: ", images);
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
  video.srcObject = stream;
	console.log('stream: ', stream);
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
		console.log("load meta success: ", video);
      resolve(video);
    };
  });
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
			//console.log('Closest Match Index : ' + closestMatchIndex);
			//console.log('Closet Match Value : ' + closestMatch);
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

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
