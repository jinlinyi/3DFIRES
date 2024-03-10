// https://newbedev.com/javascript-how-to-pass-parameter-to-event-listener-javascript-code-example


function enableInteractionTieControls() {
    var obj = document.getElementById('tie_controls')
    
    if (obj.hasAttribute('tie-interactions')){
        obj.removeAttribute('tie-interactions')
        // obj.textContent = "Tie Controls"
    }
    else{
        var att = document.createAttribute("tie-interactions");
        obj.setAttributeNode(att);
        console.log('Setting attribute')
        // obj.textContent = "UnTie Controls"
    }

}

var modelViewerClick = function(obj, evt) {
    var tie_controls = true;

    if (evt.detail.source == 'user-interaction') {
        var orbit = obj.getCameraOrbit();
        var text = obj.id;

        // Extract r_id from the ID of the clicked viewer
        var r_id = text.split("_")[1].substring(1);

        if (tie_controls) {
            // Get all model viewers in the same row
            var matchingElems = document.querySelectorAll(`[id^="glb_r${r_id}_"]`);

            matchingElems.forEach(elem => {
                if (elem.id !== text) { // Exclude the clicked viewer itself
                    elem.cameraOrbit = orbit.toString();
                    elem.interactionPrompt = 'none';
                    elem.jumpCameraToGoal();
                }
            });
        }
    }
}

function modelEventListeners(){
    console.log('enabled interactions')
    var x = document.getElementsByTagName('model-viewer');
    Array.from(x).forEach((el) => {
        var  mvid = el.getAttribute("mv-id")
        var id = el.getAttribute("id")
        el.addEventListener('camera-change', modelViewerClick.bind(event, el), 'false')
    });

}
