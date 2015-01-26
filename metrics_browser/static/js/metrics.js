// Model parameter sliders and axis radio buttons.
var params = {
    r: {init: 3.0, min: 0.1, max: 10, symbol: '&lambda;'},
    a: {init: 0.5, min: 0.1, max: 10, symbol: 'a'},
    c: {init: 1.0, min: 0.1, max: 10, symbol: 'c'},
    k: {init: 0.9, min: 0.1, max: 10, symbol: 'k'},
    mh: {init: 0.25, min: 0.1, max: 0.5, symbol: '&mu;<sub>H</sub>'},
    mp: {init: 0.25, min: 0.1, max: 0.5, symbol: '&mu;<sub>P</sub>'},
    Sh: {init: 0.02, min: 0, max: 1.0, symbol: '&Sigma;<sub>H</sub>'},
    Shh: {init: 0.01, min: 0, max: 1.0, symbol: '&Sigma;<sub>HH</sub>'},
    Sp: {init: 0.02, min: 0, max: 1.0, symbol: '&Sigma;<sub>P</sub>'},
    Spp: {init: 0.01, min: 0, max: 1.0, symbol: '&Sigma;<sub>PP</sub>'},
    im: {init: 0, min: -1, max: 1, hidden: true, symbol: 'Im(z)'},
    re: {init: 0, min: -1, max: 1, hidden: true, symbol: 'Re(z)'}
};

var metrics = {
    maxfracsync: {name: 'max fraction synchrony'},
    avgfracsync: {name: 'avg fraction synchrony'},
    xfer: {name: 'transfer function', force_axes: {x: 're', y: 'im'}}
};

var selections = {metric: 'xfer', axes: {x: 're', y: 'im'}};

var dims = {x: 20, y: 20};

function make_spec(data) {
    console.log(data.data);

    return {
        name: 'plot',
        width: 700,
        height: 700,
        padding: {top: 0, bottom: 0, left: 0, right: 0},
        data: [
            {name: 'tree', format: {type: 'treejson'}, values: data.data},
            {name: 'flattened',  source: 'tree', 'transform': [{type: 'flatten'}]}
        ],
        scales: [
            {
                name: 'spy', type: 'ordinal', range: 'height', zero: true,
                domain: {data: 'tree', field: 'data.var1'}
            }, {
                name: 'spx', type: 'ordinal', range: 'width', zero: true,
                domain: {data: 'tree', field: 'data.var2'}
            }, {
                name: 'fz', type: 'linear', range: ['#000', '#FFF'],
                domain: {data: 'flattened', field: 'data.z'}
            }
        ],
        marks: [
            {
                type: 'group',
                from: {data: 'tree'},
                properties: {
                    enter: {
                        x: {scale: 'spx', field: 'data.var2'},
                        y: {scale: 'spy', field: 'data.var1'},
                        width: {scale: 'spx', band: true},
                        height: {scale: 'spy', band: true}
                    }
                },
                /*axes: [{type: 'x', scale: 'x'}, {type: 'y', scale: 'y'}],*/
                scales: [
                    {
                        name: 'x', range: 'width', type: 'ordinal', zero: true,
                        domain: {field: 'data.x'}
                    },
                    {
                        name: 'y', range: 'height', type: 'ordinal', zero: true,
                        domain: {field: 'data.y'}
                    }
                ],
                marks: [
                    {
                        type: 'rect',
                        properties: {
                            enter: {
                                x: {scale: 'x', field: 'data.x'},
                                y: {scale: 'y', field: 'data.y'},
                                width: {scale: 'x', band: true},
                                height: {scale: 'y', band: true}
                            },
                            update: {
                                fill: {scale: 'fz', field: 'data.z'}
                            }
                        }
                    }
                ]
            }
        ]
    };
}

var view;

function draw_vis(element, data_uri) {
    $.get(data_uri, function(data) {
        view = vg.parse.spec(make_spec(data), function (chart) {
            chart({el: element}).update();
        });
    });
}

function update_form() {
    // Update axes selections and disable associated sliders.
    selections['metric'] = $('select[name=metric]').val();
    var metric = metrics[selections['metric']];

    var forced = metric.hasOwnProperty('force_axes');

    var radios = $('input:radio[data-prop=axis]');
    var x_radios = radios.filter('[data-axis=x]');
    var y_radios = radios.filter('[data-axis=y]');
    var sliders = $('div[data-prop=value]');

    console.log(selections['metric'], metric, forced);

    if (forced) {
        selections['axes'] = metric.force_axes;
    }

    var param;

    for (param in params) {
        if (params.hasOwnProperty(param)) {
            var paramsel = '[data-param=' + param + ']';
            var x_axis = x_radios.filter(paramsel).prop('checked');
            var y_axis = y_radios.filter(paramsel).prop('checked');
            var slider = sliders.filter(paramsel);
            var input = $('input[name=' + param + ']');

            input.val(slider.slider('value'));
            slider.slider({disabled: (x_axis || y_axis) && !forced});

            if (!forced) {
                if (x_axis) selections['axes']['x'] = param;
                if (y_axis) selections['axes']['y'] = param;
            }
        }
    }

    // Update values to be sent through form.
    for (var ax in selections['axes']) {
        if (selections['axes'].hasOwnProperty(ax)) {
            param = selections['axes'][ax];

            $('input[name=axis_' + ax + '_min]').val(params[param].min);
            $('input[name=axis_' + ax + '_max]').val(params[param].max);
            $('input[name=axis_' + ax + '_n]').val(dims[ax]);
            $('input[name=axis_' + ax + ']').val(param);
        }
    }
}

$(document).ready(function() {
    $('#params')
        // Sliders and axis radio buttons.
        .append(_.map(params, function(param, name) {
            return $('<div/>', {class: 'paramcontainer'})
                // Label.
                .append($('<label/>', {html: param.symbol}))
                // Slider.
                .append($('<input/>', {
                    type: 'hidden',
                    name: name,
                    value: param.init
                }))
                .append(
                    $('<div/>', {
                        'data-param': name,
                        'data-prop': 'value'
                    }).slider(param).slider({
                        step: (param.max - param.min) / 100.0,
                        value: param.init,
                        change: function(ev) {
                            update_form();

                            draw_vis(
                                '#visualizations',
                                '/pcolor.json?' + $('#params').serialize()
                            );
                        }
                    }).slider("pips", {
                        step: (param.max - param.min) / 10.0
                    }).slider("float")
                )
                // Axis radio buttons.
                .append(['x', 'y'].map(function (axis) {
                    return $('<input/>', {
                        type: 'radio',
                        'data-param': name,
                        'data-axis': axis,
                        'data-prop': 'axis',
                        name: 'radio_axis_' + axis,
                        value: name
                    }).click(update_form);
                })).css(
                    'display',
                    param.hasOwnProperty('hidden') ? 'none' : 'block'
                );
        }))
        // Select menu for metric.
        .append(['x', 'y'].map(function(axis) {
            return $('<input/>', {type: 'hidden', name: 'axis_' + axis});
        }))
        .append($('<label/>', {text: 'metric', for: 'metric'}))
        .append(
            $('<select/>', {name: 'metric', 'data-prop': 'metric'}).append(
                _.map(metrics, function(metric, key) {
                    return $('<option/>', {
                        text: metric.name,
                        value: key
                    })
                })
            ).change(update_form)
        );
});