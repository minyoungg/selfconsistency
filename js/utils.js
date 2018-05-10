
var cycle_idx = {};
var cycle_groups = {};
var cycle_ims = {};
var vid_focus = null;
 
// http://stackoverflow.com/questions/164397/javascript-how-do-i-print-a-message-to-the-error-console
function log(msg) {
  setTimeout(function() {
    throw new Error(msg);
  }, 0);
}
                 
function mod(x, modulus) {
   if (x < 0) {
      x += Math.ceil(-x/modulus)*modulus;
   }
   return x % modulus;
}
 
function cycle(group, dim, delta, fast, ask) {
//  alert("group = " + group);
//  alert("in cycle_groups " + (group in cycle_groups));
//log('cycle called');
  var exact = -1;
  if (ask) {
    exact = parseInt(prompt("Frame number", "0"))
  } else if (fast) {
    delta *= 10;
  }
  for (var i = 0; i < cycle_groups[group].length; i++) {
    var id = cycle_groups[group][i];
    var ims = cycle_ims[id];
    cycle_idx[id][dim] = mod((exact < 0 ? cycle_idx[id][dim] + delta : exact), (dim == 0 ? ims.length : ims[cycle_idx[id][0]].length));
    //log(cycle_idx[id]);
    // clamp the other index
    cycle_idx[id][1-dim] = Math.max(0, Math.min(cycle_idx[id][1-dim], (dim == 1 ? ims.length : ims[cycle_idx[id][0]].length)));
    document.getElementById(id).src = ims[cycle_idx[id][0]][cycle_idx[id][1]];
  }
}
 
 
// the row is controlled by the number keys; column by clicking
function switch_row(e, group) {
//log('sr');
  for (var i = 0; i < cycle_groups[group].length; i++) {
     var id = cycle_groups[group][i];
     var ims = cycle_ims[id];
     // 1 - 9 ==> 0 - 8;
     var s = 48; // ascii '1'
     var n = e.charCode - s - 1;
     if (0 <= n && n < ims.length) {
        if (!(id in cycle_idx)) {
           cycle_idx[id] = [0, 0];
        }
        cycle_idx[id][0] = n;
        cycle_idx[id][1] = mod(cycle_idx[id][1], ims[n].length);
        document.getElementById(id).src = ims[cycle_idx[id][0]][cycle_idx[id][1]];
     }
   }
}
 
var curr_cycle_group = null;
//document.onkeypress = function(e) {
//  if (curr_cycle_group != null) {
//    switch_row(e, curr_cycle_group);
//  }
//}
 
document.onkeydown = function (e) {
 if (vid_focus != null && vid_focus.paused) {
    // rough estimate
    var frame_duration = 1./30;
    var vf = vid_focus;
     // 37 39 are left/right
    if (String.fromCharCode(e.keyCode) == 'O') {
      vf.currentTime = Math.max(vf.currentTime - frame_duration, 0);
    } else if (String.fromCharCode(e.keyCode) == 'P') {
      vf.currentTime = Math.min(vf.currentTime + frame_duration, vf.duration);
    }
    return true;
  } else if (curr_cycle_group != null) {
    switch_row(e, curr_cycle_group);
  }
};
 
 
function register_cycle(group, id, ims, start) {
  if (!(id in cycle_ims)) {
    if (!(group in cycle_groups)) {
      cycle_groups[group] = [];
    }
    cycle_groups[group].push(id);
    cycle_ims[id] = ims;
    cycle_idx[id] = start;
  }
}
function getParameterByName(name) {
  name = name.replace(/[\[]/, "\\[").replace(/[\]]/, "\\]");
  var regex = new RegExp("[\?&]" + name + "=([^&#]*)");
  var results = regex.exec(location.search);
  return results == null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
}
 